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
        if let Some(result) =
            crate::simd_sumcheck::dispatch::try_simd_dispatch::<BF, EF>(evaluations, transcript)
        {
            return result;
        }
        // Extension field dispatch (BF == EF == Goldilocks ext2/ext3)
        if let Some(result) =
            crate::simd_sumcheck::dispatch::try_simd_ext_dispatch::<BF, EF>(
                evaluations,
                transcript,
            )
        {
            return result;
        }
    }

    let num_rounds = evaluations.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = vec![];
    let mut verifier_messages: Vec<EF> = vec![];

    // ── Round 0: evaluate in BF, lift to EF, cross-field reduce ──
    if num_rounds > 0 {
        let msg_bf = pairwise::evaluate(evaluations);
        let msg = (EF::from(msg_bf.0), EF::from(msg_bf.1));

        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        let chg = transcript.read();
        verifier_messages.push(chg);

        // Cross-field reduce: BF evaluations + EF challenge → Vec<EF>
        let mut ef_evals = pairwise::cross_field_reduce(evaluations, chg);

        // Remaining rounds work in EF
        for _ in 1..num_rounds {
            // Try SIMD extension evaluate (accelerates when EF is Goldilocks-based)
            #[cfg(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            ))]
            let msg = crate::simd_sumcheck::dispatch::try_simd_ext_evaluate(&ef_evals)
                .unwrap_or_else(|| pairwise::evaluate(&ef_evals));

            #[cfg(not(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            )))]
            let msg = pairwise::evaluate(&ef_evals);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            let chg = transcript.read();
            verifier_messages.push(chg);

            // Try SIMD extension reduce (accelerates when EF is Goldilocks-based)
            #[cfg(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            ))]
            let reduced =
                crate::simd_sumcheck::dispatch::try_simd_ext_reduce(&mut ef_evals, chg);

            #[cfg(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            ))]
            if !reduced {
                pairwise::reduce_evaluations(&mut ef_evals, chg);
            }

            #[cfg(not(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            )))]
            pairwise::reduce_evaluations(&mut ef_evals, chg);
        }
    }

    Sumcheck {
        verifier_messages,
        prover_messages,
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
}
