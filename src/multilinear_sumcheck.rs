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
use crate::simd_fields::SimdAccelerated;
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
    #[cfg(target_arch = "aarch64")]
    if let Some(result) = try_simd_dispatch::<BF, EF>(evaluations, transcript) {
        return result;
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
            let msg = pairwise::evaluate(&ef_evals);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            let chg = transcript.read();
            verifier_messages.push(chg);

            pairwise::reduce_evaluations(&mut ef_evals, chg);
        }
    }

    Sumcheck {
        verifier_messages,
        prover_messages,
    }
}

/// Try to dispatch to the SIMD backend when `BF == EF` and `BF` is a known
/// SIMD-accelerated type (currently: Goldilocks F64).
///
/// Returns `Some(result)` if the SIMD path was taken, `None` otherwise.
///
/// In monomorphized code, the `TypeId` checks are compile-time constants.
/// LLVM eliminates the entire function body for non-matching types — zero cost.
#[cfg(target_arch = "aarch64")]
fn try_simd_dispatch<BF: Field, EF: Field + From<BF>>(
    evaluations: &mut [BF],
    transcript: &mut impl Transcript<EF>,
) -> Option<Sumcheck<EF>> {
    use crate::tests::F64;
    use std::any::TypeId;

    // Both checks are compile-time constants in monomorphized code.
    if TypeId::of::<BF>() == TypeId::of::<EF>() && TypeId::of::<BF>() == TypeId::of::<F64>() {
        // BF == EF == F64 (verified via TypeId).

        // Cast &mut [BF] → &[F64] (same type, same layout).
        let evals_f64: &[F64] = unsafe {
            core::slice::from_raw_parts(evaluations.as_ptr() as *const F64, evaluations.len())
        };

        // Single closure for transcript round-step: write (s0, s1), return challenge.
        // This avoids the double-mutable-borrow issue with separate write/read closures.
        let result_f64 = simd_sumcheck_raw_f64(evals_f64, |s0, s1| {
            // SAFETY: EF == F64, so the in-memory representation is identical.
            let s0_ef: EF = unsafe { core::mem::transmute_copy(&s0) };
            let s1_ef: EF = unsafe { core::mem::transmute_copy(&s1) };
            transcript.write(s0_ef);
            transcript.write(s1_ef);
            let chg_ef: EF = transcript.read();
            unsafe { core::mem::transmute_copy(&chg_ef) }
        });

        // Cast Sumcheck<F64> → Sumcheck<EF>.
        // SAFETY: F64 == EF (verified above), so layout is identical.
        let result: Sumcheck<EF> = Sumcheck {
            verifier_messages: unsafe { core::mem::transmute(result_f64.verifier_messages) },
            prover_messages: unsafe { core::mem::transmute(result_f64.prover_messages) },
        };

        return Some(result);
    }

    None
}

/// Raw SIMD sumcheck for F64, using a single closure for transcript interaction.
///
/// `round_step(s0, s1) -> challenge`: Writes the round messages to the transcript
/// and returns the verifier's challenge. This single-closure design avoids borrowing
/// issues with the outer transcript reference.
#[cfg(target_arch = "aarch64")]
fn simd_sumcheck_raw_f64(
    evaluations: &[crate::tests::F64],
    mut round_step: impl FnMut(crate::tests::F64, crate::tests::F64) -> crate::tests::F64,
) -> Sumcheck<crate::tests::F64> {
    use crate::simd_fields::SimdAccelerated;
    use crate::tests::F64;

    let num_rounds = evaluations.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(F64, F64)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<F64> = Vec::with_capacity(num_rounds);

    let mut buf = F64::slice_to_raw(evaluations);
    let mut active_len = buf.len();

    for round in 0..num_rounds {
        let half = active_len / 2;

        let (s0, s1) = eval_raw::<<F64 as SimdAccelerated>::Backend>(&buf[..active_len]);

        let msg_s0 = F64::from_raw(s0);
        let msg_s1 = F64::from_raw(s1);

        prover_messages.push((msg_s0, msg_s1));
        let challenge = round_step(msg_s0, msg_s1);
        verifier_messages.push(challenge);

        if round < num_rounds - 1 {
            reduce_raw::<<F64 as SimdAccelerated>::Backend>(&mut buf, half, F64::to_raw(challenge));
            active_len = half;
        }
    }

    Sumcheck {
        verifier_messages,
        prover_messages,
    }
}

/// SIMD-accelerated multilinear sumcheck (base = extension).
///
/// Same semantics as [`multilinear_sumcheck`], but uses native SIMD intrinsics
/// for the hot-path evaluate and reduce operations. The dispatch is **compile-time**:
/// this function only exists for fields that implement [`SimdAccelerated`].
///
/// # How it works
///
/// 1. Converts evaluations from arkworks `Field` representation to raw scalars (O(n))
/// 2. Runs the sumcheck entirely in the raw SIMD domain (O(n log n))
/// 3. Wraps the round messages back in arkworks types
///
/// The O(n) conversion cost is amortized by the O(n log n) sumcheck.
///
/// # Usage
///
/// ```text
/// // This compiles only if F64 implements SimdAccelerated:
/// let result = simd_multilinear_sumcheck::<F64>(&evals, &mut transcript);
/// ```
pub fn simd_multilinear_sumcheck<BF>(
    evaluations: &[BF],
    transcript: &mut impl Transcript<BF>,
) -> Sumcheck<BF>
where
    BF: Field + SimdAccelerated,
{
    assert!(
        evaluations.len().count_ones() == 1,
        "length must be a power of 2"
    );
    assert!(evaluations.len() >= 2, "need at least 1 variable");

    let num_rounds = evaluations.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(BF, BF)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<BF> = Vec::with_capacity(num_rounds);

    // Copy to raw scalars — zero-cost memcpy for Montgomery-form types.
    let mut buf = BF::slice_to_raw(evaluations);
    let mut active_len = buf.len();

    for round in 0..num_rounds {
        let half = active_len / 2;

        // ── Evaluate: sum even-indexed and odd-indexed elements ──
        let (s0, s1) = eval_raw::<BF::Backend>(&buf[..active_len]);

        let msg_s0 = BF::from_raw(s0);
        let msg_s1 = BF::from_raw(s1);

        prover_messages.push((msg_s0, msg_s1));
        transcript.write(msg_s0);
        transcript.write(msg_s1);

        let challenge = transcript.read();
        verifier_messages.push(challenge);

        // ── Reduce in-place ──
        if round < num_rounds - 1 {
            reduce_raw::<BF::Backend>(&mut buf, half, BF::to_raw(challenge));
            active_len = half;
        }
    }

    Sumcheck {
        verifier_messages,
        prover_messages,
    }
}

/// Below this element count, stay single-threaded (rayon spawn overhead dominates).
/// Above it, parallelize evaluate & reduce. 128K elements ≈ 2^17.
const PAR_THRESHOLD: usize = 1 << 17;

/// Sum even-indexed and odd-indexed elements of a raw scalar slice.
#[inline(always)]
fn eval_raw<F: crate::simd_fields::SimdBaseField>(evals: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    #[cfg(feature = "parallel")]
    {
        if evals.len() >= PAR_THRESHOLD {
            return eval_raw_parallel::<F>(evals);
        }
    }
    eval_raw_seq::<F>(evals)
}

/// Sequential evaluate.
#[inline(always)]
fn eval_raw_seq<F: crate::simd_fields::SimdBaseField>(
    evals: &[F::Scalar],
) -> (F::Scalar, F::Scalar) {
    let mut s0 = F::ZERO;
    let mut s1 = F::ZERO;
    let mut i = 0;
    while i + 1 < evals.len() {
        s0 = F::scalar_add(s0, evals[i]);
        s1 = F::scalar_add(s1, evals[i + 1]);
        i += 2;
    }
    (s0, s1)
}

/// Parallel evaluate using rayon.
#[cfg(feature = "parallel")]
fn eval_raw_parallel<F: crate::simd_fields::SimdBaseField>(
    evals: &[F::Scalar],
) -> (F::Scalar, F::Scalar) {
    use rayon::prelude::*;

    // Split into chunks of pairs, compute partial sums in parallel, then merge.
    let chunk_pairs = 16_384; // pairs per chunk
    let chunk_scalars = chunk_pairs * 2;

    let (s0, s1) = evals
        .par_chunks(chunk_scalars)
        .map(|chunk| eval_raw_seq::<F>(chunk))
        .reduce(
            || (F::ZERO, F::ZERO),
            |(a0, a1), (b0, b1)| (F::scalar_add(a0, b0), F::scalar_add(a1, b1)),
        );
    (s0, s1)
}

/// In-place pairwise reduce: `buf[i] = buf[2i] + c * (buf[2i+1] - buf[2i])`.
#[inline(always)]
fn reduce_raw<F: crate::simd_fields::SimdBaseField>(
    buf: &mut [F::Scalar],
    half: usize,
    c: F::Scalar,
) {
    #[cfg(feature = "parallel")]
    {
        if half >= PAR_THRESHOLD / 2 {
            reduce_raw_parallel::<F>(buf, half, c);
            return;
        }
    }
    reduce_raw_seq::<F>(buf, half, c);
}

/// Sequential reduce.
#[inline(always)]
fn reduce_raw_seq<F: crate::simd_fields::SimdBaseField>(
    buf: &mut [F::Scalar],
    half: usize,
    c: F::Scalar,
) {
    for i in 0..half {
        let a = buf[2 * i];
        let b = buf[2 * i + 1];
        let diff = F::scalar_sub(b, a);
        let scaled = F::scalar_mul(c, diff);
        buf[i] = F::scalar_add(a, scaled);
    }
}

/// Parallel reduce using rayon.
///
/// Strategy: we can't trivially do in-place parallel reduce because of
/// aliasing (buf[i] reads from buf[2i]). Instead, we first compute
/// the reduced values into a temporary buffer in parallel, then copy back.
#[cfg(feature = "parallel")]
fn reduce_raw_parallel<F: crate::simd_fields::SimdBaseField>(
    buf: &mut [F::Scalar],
    half: usize,
    c: F::Scalar,
) {
    use rayon::prelude::*;

    // Compute reduced values in parallel from the pairs region.
    let pairs = &buf[..2 * half];
    let reduced: Vec<F::Scalar> = pairs
        .par_chunks(2)
        .map(|pair| {
            let a = pair[0];
            let b = pair[1];
            let diff = F::scalar_sub(b, a);
            let scaled = F::scalar_mul(c, diff);
            F::scalar_add(a, scaled)
        })
        .collect();

    // Copy back into the first `half` positions.
    buf[..half].copy_from_slice(&reduced);
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

        // Run SIMD sumcheck with the same transcript seeding
        let mut rng2 = test_rng();
        let mut transcript2 = SanityTranscript::new(&mut rng2);
        let simd_result = simd_multilinear_sumcheck::<F64>(&evaluations, &mut transcript2);

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
}
