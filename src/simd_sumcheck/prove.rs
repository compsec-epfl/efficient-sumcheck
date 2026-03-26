//! SIMD-vectorized multilinear sumcheck prover (base = extension).
//!
//! This is the base=extension (EXT_DEGREE=1) sumcheck: the entire protocol
//! stays in the base field, no extension promotion or Karatsuba needed.

use crate::simd_fields::SimdBaseField;
use crate::simd_sumcheck::evaluate::evaluate_parallel;
use crate::simd_sumcheck::reduce::reduce_parallel;

/// Result of the SIMD multilinear sumcheck.
#[derive(Debug)]
pub struct SimdSumcheck<S: Copy> {
    /// Round messages: `(s(0), s(1))` for each round.
    pub prover_messages: Vec<(S, S)>,
    /// Verifier challenges, one per round (except the last).
    pub verifier_messages: Vec<S>,
}

/// Run the SIMD multilinear sumcheck (base = extension).
///
/// `evals` are the raw scalar evaluations of the multilinear polynomial on the
/// boolean hypercube. `challenge_fn` provides the verifier's challenge after each
/// round (e.g., from a Fiat-Shamir transcript).
///
/// This function consumes the evaluations and runs the full sumcheck protocol,
/// returning the transcript.
pub fn prove_base_eq_ext<F: SimdBaseField>(
    evals: &[F::Scalar],
    mut challenge_fn: impl FnMut(F::Scalar, F::Scalar) -> F::Scalar,
) -> SimdSumcheck<F::Scalar> {
    assert!(
        evals.len().count_ones() == 1 && evals.len() >= 2,
        "evals length must be a power of 2 and >= 2"
    );

    let num_rounds = evals.len().trailing_zeros() as usize;
    let mut prover_messages = Vec::with_capacity(num_rounds);
    let mut verifier_messages = Vec::with_capacity(num_rounds);

    let mut current = evals.to_vec();

    for round in 0..num_rounds {
        // Evaluate: sum even-indexed and odd-indexed elements
        let (s0, s1) = evaluate_parallel::<F>(&current);
        prover_messages.push((s0, s1));

        if round < num_rounds - 1 {
            // Get verifier challenge
            let challenge = challenge_fn(s0, s1);
            verifier_messages.push(challenge);

            // Reduce
            current = reduce_parallel::<F>(&current, challenge);
        }
    }

    SimdSumcheck {
        prover_messages,
        verifier_messages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multilinear_sumcheck;
    use crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::{PrimeField, UniformRand};
    use ark_std::test_rng;

    fn to_raw(f: F64) -> u64 {
        f.into_bigint().0[0]
    }

    #[test]
    fn test_simd_sumcheck_matches_reference() {
        let num_vars = 16;
        let n = 1 << num_vars;

        let mut rng = test_rng();
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_raw(*f)).collect();

        // Run the reference sumcheck
        let mut ref_evals = evals_ff.clone();
        let mut ref_rng = test_rng();
        let mut ref_transcript = SanityTranscript::new(&mut ref_rng);
        let ref_result = multilinear_sumcheck::<F64, F64>(&mut ref_evals, &mut ref_transcript);

        // Run the SIMD sumcheck with the same challenges
        // We need to produce the same challenges. The SanityTranscript uses
        // random challenges that depend on the prover messages via write/read.
        // To make this deterministic, we use the reference challenges directly.
        let ref_challenges = ref_result.verifier_messages.clone();
        let mut challenge_idx = 0;

        let simd_result = prove_base_eq_ext::<GoldilocksNeon>(&evals_raw, |_s0, _s1| {
            let c = to_raw(ref_challenges[challenge_idx]);
            challenge_idx += 1;
            c
        });

        // Check prover messages match
        assert_eq!(
            ref_result.prover_messages.len(),
            simd_result.prover_messages.len(),
            "round count mismatch"
        );

        for (i, (ref_msg, simd_msg)) in ref_result
            .prover_messages
            .iter()
            .zip(simd_result.prover_messages.iter())
            .enumerate()
        {
            assert_eq!(to_raw(ref_msg.0), simd_msg.0, "s0 mismatch at round {}", i);
            assert_eq!(to_raw(ref_msg.1), simd_msg.1, "s1 mismatch at round {}", i);
        }
    }

    #[test]
    fn test_simd_sumcheck_small() {
        // Small test (4 elements = 2 rounds)
        let evals_raw: Vec<u64> = vec![1, 2, 3, 4];
        // sum = 10, s0 = 1+3=4, s1 = 2+4=6

        let simd_result = prove_base_eq_ext::<GoldilocksNeon>(
            &evals_raw,
            |_s0, _s1| 7, // fixed challenge
        );

        assert_eq!(simd_result.prover_messages.len(), 2);
        assert_eq!(simd_result.verifier_messages.len(), 1);

        // Round 0: s0 = 4, s1 = 6
        assert_eq!(simd_result.prover_messages[0], (4, 6));

        // After reduce with challenge=7: for each pair (a, b):
        //   a + 7*(b-a) = a + 7b - 7a = 7b - 6a
        //   pair (1,2): 1 + 7*(2-1) = 8
        //   pair (3,4): 3 + 7*(4-3) = 10
        // Round 1: s0 = 8, s1 = 10
        assert_eq!(simd_result.prover_messages[1], (8, 10));
    }
}
