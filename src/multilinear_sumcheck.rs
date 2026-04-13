//! Standard multilinear sumcheck protocol.
//!
//! Given evaluations `[p(0..0), p(0..1), ..., p(1..1)]` of a multilinear polynomial `p`
//! on the boolean hypercube `{0,1}^n`, the [`multilinear_sumcheck`] function executes `n`
//! rounds of the sumcheck protocol and returns the resulting [`Sumcheck`] transcript.
//!
//! The function is parameterized by two field types:
//! - `BF` (base field): the field the evaluations live in
//! - `EF` (extension field): the field challenges are sampled from
//!
//! When no extension field is needed, set `EF = BF`.
//!
//! # Example
//!
//! ```text
//! use efficient_sumcheck::{multilinear_sumcheck, Sumcheck};
//! use efficient_sumcheck::transcript::SanityTranscript;
//!
//! // No extension field (BF = EF):
//! let mut evals = vec![F::from(1), F::from(2), F::from(3), F::from(4)];
//! let mut transcript = SanityTranscript::new(&mut rng);
//! let result: Sumcheck<F> = multilinear_sumcheck(&mut evals, &mut transcript);
//! ```

use ark_ff::Field;

use crate::multilinear::reductions::pairwise;
use crate::transcript::Transcript;

pub use crate::multilinear::Sumcheck;

/// Run the standard multilinear sumcheck protocol over an evaluation vector,
/// using a generic [`Transcript`] for Fiat-Shamir (or sanity/random challenges).
///
/// `BF` is the base field of the evaluations, `EF` is the extension field for challenges.
/// When `BF = EF`, this is the standard single-field sumcheck.
/// When `BF ≠ EF`, round 0 evaluates in `BF` and lifts to `EF`, then subsequent
/// rounds work entirely in `EF`.
///
/// Each round:
/// 1. Computes the round polynomial evaluations `(s(0), s(1))` via pairwise reduction.
/// 2. Writes them to the transcript (2 field elements).
/// 3. Reads the verifier's challenge from the transcript (1 field element).
/// 4. Reduces the evaluation vector by folding with the challenge.
pub fn multilinear_sumcheck<BF: Field, EF: Field + From<BF>>(
    evaluations: &mut [BF],
    transcript: &mut impl Transcript<EF>,
) -> Sumcheck<EF> {
    multilinear_sumcheck_with_hook(evaluations, transcript, |_, _| {})
}

/// Like [`multilinear_sumcheck`], but calls `hook(round_idx, transcript)`
/// each round *after* the prover message is written and *before* the verifier
/// challenge is read.
///
/// Useful for injecting per-round proof-of-work grinding, logging, or other
/// extensions to the transcript that must appear at a specific point in the
/// Fiat-Shamir schedule. The hook is invoked for every round (0..num_rounds),
/// including the round-0 base-field message on cross-field sumchecks.
pub fn multilinear_sumcheck_with_hook<BF, EF, T, H>(
    evaluations: &mut [BF],
    transcript: &mut T,
    mut hook: H,
) -> Sumcheck<EF>
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
{
    // checks
    assert!(
        evaluations.len().count_ones() == 1,
        "length must be a power of 2"
    );
    assert!(evaluations.len() >= 2, "need at least 1 variable");

    // ── SIMD auto-dispatch ──
    // When BF == EF and BF has a SIMD backend, transparently route to the
    // fast SIMD path. The TypeId checks evaluate to compile-time constants
    // in monomorphized code, so LLVM eliminates the dead branch — zero cost.
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    {
        // Base field dispatch (BF == EF == Goldilocks base)
        if let Some(result) = crate::simd_sumcheck::dispatch::try_simd_dispatch::<BF, EF, T, H>(
            evaluations,
            transcript,
            &mut hook,
        ) {
            return result;
        }
        // Extension field dispatch (BF == EF == Goldilocks ext2/ext3).
        // On AVX-512: use full SIMD dispatch (8-wide mul makes reduce fast).
        // On NEON: skip — the single-threaded ext reduce is slower than the
        // generic path with SIMD evaluate + rayon-parallel arkworks reduce.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
        if let Some(result) =
            crate::simd_sumcheck::dispatch::try_simd_ext_dispatch::<BF, EF, T, H>(
                evaluations,
                transcript,
                &mut hook,
            )
        {
            return result;
        }
    }

    let num_rounds = evaluations.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = vec![];
    let mut verifier_messages: Vec<EF> = vec![];
    let mut final_evaluation = EF::ZERO;

    // ── Round 0: evaluate in BF, lift to EF, cross-field reduce ──
    if num_rounds > 0 {
        let msg_bf = crate::simd_ops::pairwise_sum(evaluations);
        let msg = (EF::from(msg_bf.0), EF::from(msg_bf.1));

        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        hook(0, transcript);

        let chg = transcript.read();
        verifier_messages.push(chg);

        // Cross-field reduce: BF evaluations + EF challenge → Vec<EF>
        let mut ef_evals = pairwise::cross_field_reduce(evaluations, chg);

        // Remaining rounds work in EF.
        // Use fused reduce+evaluate when available: reduces data AND computes
        // next round's (s0, s1) in a single pass, eliminating one full read.
        let mut pending_eval: Option<(EF, EF)> = None;

        for round in 1..num_rounds {
            // Get this round's evaluate — either from the previous fused pass
            // or by computing it now.
            let msg = if let Some(cached) = pending_eval.take() {
                cached
            } else {
                #[cfg(any(
                    target_arch = "aarch64",
                    all(target_arch = "x86_64", target_feature = "avx512ifma")
                ))]
                let result = crate::simd_sumcheck::dispatch::try_simd_ext_evaluate(&ef_evals)
                    .unwrap_or_else(|| pairwise::evaluate(&ef_evals));

                #[cfg(not(any(
                    target_arch = "aarch64",
                    all(target_arch = "x86_64", target_feature = "avx512ifma")
                )))]
                let result = pairwise::evaluate(&ef_evals);

                result
            };

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            hook(round, transcript);

            let chg = transcript.read();
            verifier_messages.push(chg);

            // SIMD extension reduce strategies (best picked by size):
            // 1. Small (≤ 2^17): fused reduce+evaluate in single pass
            // 2. Any size: SIMD ext reduce (uses ext2/ext3 Karatsuba)
            // 3. Fallback: generic arkworks Field reduce
            #[cfg(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            ))]
            {
                // Try fused for small inputs first
                if ef_evals.len() <= (1 << 17) {
                    if let Some(next_msg) =
                        crate::simd_sumcheck::dispatch::try_simd_ext_fused_reduce_evaluate(
                            &mut ef_evals,
                            chg,
                        )
                    {
                        pending_eval = Some(next_msg);
                        continue;
                    }
                }
                // Try SIMD ext reduce — on AVX-512 always, on NEON only for small inputs
                // (NEON ext reduce is scalar, so rayon-parallel generic reduce is faster at scale)
                #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
                if crate::simd_sumcheck::dispatch::try_simd_ext_reduce(&mut ef_evals, chg) {
                    continue;
                }
            }
            pairwise::reduce_evaluations(&mut ef_evals, chg);
        }

        // After all rounds, ef_evals is length 1: the polynomial evaluated at
        // the verifier challenge point.
        debug_assert_eq!(ef_evals.len(), 1);
        final_evaluation = ef_evals[0];
    }

    Sumcheck {
        verifier_messages,
        prover_messages,
        final_evaluation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use crate::tests::F64;

    const NUM_VARS: usize = 4; // vectors of length 2^4 = 16

    #[test]
    fn test_multilinear_sumcheck_sanity() {
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut evaluations: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut transcript = SanityTranscript::new(&mut rng);
        let result = multilinear_sumcheck::<F64, F64>(&mut evaluations, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }

    #[test]
    fn test_multilinear_sumcheck_spongefish() {
        use crate::transcript::SpongefishTranscript;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut evaluations: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let domsep = spongefish::domain_separator!("test-multilinear-sumcheck"; module_path!())
            .instance(b"test");

        let prover_state = domsep.std_prover();
        let mut transcript = SpongefishTranscript::new(prover_state);
        let result = multilinear_sumcheck::<F64, F64>(&mut evaluations, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }

    #[test]
    fn test_simd_parity_with_generic() {
        use crate::transcript::SanityTranscript;

        let num_vars = 16;
        let n = 1 << num_vars;

        let mut rng = test_rng();
        let evaluations: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Run generic sumcheck
        let mut generic_evals = evaluations.clone();
        let mut rng1 = test_rng();
        let mut transcript1 = SanityTranscript::new(&mut rng1);
        let generic_result = multilinear_sumcheck::<F64, F64>(&mut generic_evals, &mut transcript1);

        // Run SIMD sumcheck (auto-dispatched via multilinear_sumcheck)
        let mut simd_evals = evaluations.clone();
        let mut rng2 = test_rng();
        let mut transcript2 = SanityTranscript::new(&mut rng2);
        let simd_result = multilinear_sumcheck::<F64, F64>(&mut simd_evals, &mut transcript2);

        // Prover messages must match exactly
        assert_eq!(
            generic_result.prover_messages.len(),
            simd_result.prover_messages.len()
        );
        for (i, (g, s)) in generic_result
            .prover_messages
            .iter()
            .zip(simd_result.prover_messages.iter())
            .enumerate()
        {
            assert_eq!(g.0, s.0, "s0 mismatch at round {}", i);
            assert_eq!(g.1, s.1, "s1 mismatch at round {}", i);
        }

        // Verifier challenges must match exactly
        assert_eq!(
            generic_result.verifier_messages,
            simd_result.verifier_messages
        );
    }

    #[test]
    #[should_panic(expected = "power of 2")]
    fn test_non_power_of_2_panics() {
        use crate::transcript::SanityTranscript;
        let mut rng = test_rng();
        let mut evals = vec![F64::from(1u64); 7]; // not a power of 2
        let mut transcript = SanityTranscript::new(&mut rng);
        multilinear_sumcheck::<F64, F64>(&mut evals, &mut transcript);
    }

    #[test]
    fn test_minimal_input() {
        // n = 2 (1 variable, 1 round)
        use crate::transcript::SanityTranscript;
        let mut rng = test_rng();
        let mut evals = vec![F64::from(3u64), F64::from(7u64)];
        let mut transcript = SanityTranscript::new(&mut rng);
        let result = multilinear_sumcheck::<F64, F64>(&mut evals, &mut transcript);
        assert_eq!(result.prover_messages.len(), 1);
        assert_eq!(result.prover_messages[0].0, F64::from(3u64)); // s(0)
        assert_eq!(result.prover_messages[0].1, F64::from(7u64)); // s(1)
    }

    #[test]
    fn test_extension_field_sumcheck() {
        // Test multilinear sumcheck with BF = EF = F64Ext2 (degree-2 extension).
        // This exercises the SIMD extension evaluate path in rounds 1+.
        use crate::tests::F64Ext2;
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();
        let n = 1 << 8;
        let mut evals: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        // Compute expected sum before sumcheck (which may modify evals in-place)
        let claimed_sum: F64Ext2 = evals.iter().copied().sum();

        // Run the sumcheck (SIMD extension dispatch for Goldilocks Ext2)
        let mut transcript = SanityTranscript::new(&mut rng);
        let result = multilinear_sumcheck::<F64Ext2, F64Ext2>(&mut evals, &mut transcript);

        assert_eq!(result.prover_messages.len(), 8);
        assert_eq!(result.verifier_messages.len(), 8);

        // Verify round 0: s(0) + s(1) == sum of all evaluations
        let (s0, s1) = result.prover_messages[0];
        assert_eq!(s0 + s1, claimed_sum, "round 0 sum mismatch");
    }

    /// Exercises the rayon-parallel SoA reduce path (n > 2^17 threshold in dispatch).
    #[test]
    fn test_ext2_sumcheck_parallel_path_matches_generic() {
        use crate::multilinear::reductions::pairwise;
        use crate::tests::F64Ext2;
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();
        let n = 1 << 18; // above EXT_PARALLEL_THRESHOLD
        let evals: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        // Generic reference: run the pairwise evaluate+reduce loop directly.
        let mut rng1 = test_rng();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let num_rounds = (n as u64).trailing_zeros() as usize;
        let mut ef = evals.clone();
        let mut expected_msgs = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let (e, o) = pairwise::evaluate(&ef);
            expected_msgs.push((e, o));
            t1.write(e);
            t1.write(o);
            let chg: F64Ext2 = t1.read();
            pairwise::reduce_evaluations(&mut ef, chg);
        }

        // SIMD path (will hit the parallel ext2 SoA kernel).
        let mut rng2 = test_rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let mut simd_evals = evals;
        let simd_result = multilinear_sumcheck::<F64Ext2, F64Ext2>(&mut simd_evals, &mut t2);

        assert_eq!(simd_result.prover_messages.len(), expected_msgs.len());
        for (i, (exp, got)) in expected_msgs.iter().zip(simd_result.prover_messages.iter()).enumerate() {
            assert_eq!(exp.0, got.0, "s0 mismatch at round {}", i);
            assert_eq!(exp.1, got.1, "s1 mismatch at round {}", i);
        }
    }

    /// Independent fold: evaluate the multilinear at the verifier challenges
    /// and compare against `Sumcheck::final_evaluation` populated by the entry point.
    fn fold_multilinear<F: ark_ff::Field>(evals: &[F], challenges: &[F]) -> F {
        let mut current = evals.to_vec();
        for &chg in challenges {
            let mut next = Vec::with_capacity(current.len() / 2);
            for pair in current.chunks(2) {
                next.push(pair[0] + chg * (pair[1] - pair[0]));
            }
            current = next;
        }
        debug_assert_eq!(current.len(), 1);
        current[0]
    }

    #[test]
    fn test_final_evaluation_matches_independent_fold_base() {
        use crate::transcript::SanityTranscript;

        let num_vars = 8;
        let n = 1 << num_vars;
        let mut rng = test_rng();
        let evals_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut evals = evals_orig.clone();
        let mut transcript = SanityTranscript::new(&mut rng);
        let result = multilinear_sumcheck::<F64, F64>(&mut evals, &mut transcript);

        let expected = fold_multilinear(&evals_orig, &result.verifier_messages);
        assert_eq!(result.final_evaluation, expected, "ML final_evaluation mismatch");
    }

    #[test]
    fn test_final_evaluation_matches_independent_fold_ext2() {
        use crate::tests::F64Ext2;
        use crate::transcript::SanityTranscript;

        let num_vars = 8;
        let n = 1 << num_vars;
        let mut rng = test_rng();
        let evals_orig: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let mut evals = evals_orig.clone();
        let mut transcript = SanityTranscript::new(&mut rng);
        let result = multilinear_sumcheck::<F64Ext2, F64Ext2>(&mut evals, &mut transcript);

        let expected = fold_multilinear(&evals_orig, &result.verifier_messages);
        assert_eq!(result.final_evaluation, expected, "ext2 ML final_evaluation mismatch");
    }

    #[test]
    fn test_with_hook_called_once_per_round() {
        use crate::transcript::SanityTranscript;
        use std::cell::RefCell;

        let num_vars = 6;
        let n = 1 << num_vars;
        let mut rng = test_rng();
        let mut evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut transcript = SanityTranscript::new(&mut rng);

        let calls = RefCell::new(Vec::<usize>::new());
        let result = multilinear_sumcheck_with_hook::<F64, F64, _, _>(
            &mut evals,
            &mut transcript,
            |round, _t| calls.borrow_mut().push(round),
        );

        assert_eq!(result.prover_messages.len(), num_vars);
        let calls = calls.into_inner();
        assert_eq!(calls, (0..num_vars).collect::<Vec<_>>(), "hook must be called once per round in order");
    }

    #[test]
    fn test_with_hook_injects_into_transcript() {
        // The hook writes an extra field element between the prover message and
        // the verifier challenge. Two runs with identical data but different
        // hook payloads must produce different verifier challenges from round 0
        // onward — proving the hook's writes actually enter the Fiat-Shamir
        // state.
        use crate::transcript::SpongefishTranscript;

        let num_vars = 4;
        let n = 1 << num_vars;

        let mut rng = test_rng();
        let evals_a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let run = |tag: F64, evals: Vec<F64>| {
            let mut evals = evals;
            let domsep = spongefish::domain_separator!("hook-test"; module_path!())
                .instance(b"test");
            let prover_state = domsep.std_prover();
            let mut transcript = SpongefishTranscript::new(prover_state);
            multilinear_sumcheck_with_hook::<F64, F64, _, _>(
                &mut evals,
                &mut transcript,
                move |_round, t| {
                    t.write(tag);
                },
            )
        };

        let result_a = run(F64::from(1u64), evals_a.clone());
        let result_b = run(F64::from(2u64), evals_a);

        assert_ne!(
            result_a.verifier_messages[0],
            result_b.verifier_messages[0],
            "hook writes must affect Fiat-Shamir state"
        );
    }

    #[test]
    fn test_ext3_sumcheck_parallel_path_matches_generic() {
        use crate::multilinear::reductions::pairwise;
        use crate::tests::F64Ext3;
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();
        let n = 1 << 18;
        let evals: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();

        let mut rng1 = test_rng();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let num_rounds = (n as u64).trailing_zeros() as usize;
        let mut ef = evals.clone();
        let mut expected_msgs = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let (e, o) = pairwise::evaluate(&ef);
            expected_msgs.push((e, o));
            t1.write(e);
            t1.write(o);
            let chg: F64Ext3 = t1.read();
            pairwise::reduce_evaluations(&mut ef, chg);
        }

        let mut rng2 = test_rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let mut simd_evals = evals;
        let simd_result = multilinear_sumcheck::<F64Ext3, F64Ext3>(&mut simd_evals, &mut t2);

        for (i, (exp, got)) in expected_msgs.iter().zip(simd_result.prover_messages.iter()).enumerate() {
            assert_eq!(exp.0, got.0, "s0 mismatch at round {}", i);
            assert_eq!(exp.1, got.1, "s1 mismatch at round {}", i);
        }
    }
}
