//! SIMD-vectorized multilinear sumcheck prover (base = extension).
//!
//! This is the base=extension (EXT_DEGREE=1) sumcheck: the entire protocol
//! stays in the base field, no extension promotion or Karatsuba needed.

use crate::simd_fields::SimdBaseField;
use crate::simd_sumcheck::evaluate::evaluate_parallel;
use crate::simd_sumcheck::reduce::reduce_parallel;

/// Result of the SIMD multilinear sumcheck over raw scalars.
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
        let (s0, s1) = evaluate_parallel::<F>(&current);
        prover_messages.push((s0, s1));

        if round < num_rounds - 1 {
            let challenge = challenge_fn(s0, s1);
            verifier_messages.push(challenge);
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
    use crate::simd_fields::goldilocks::mont_neon::MontGoldilocksNeon;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    fn to_mont(f: F64) -> u64 {
        f.value
    }

    #[test]
    fn test_simd_sumcheck_matches_reference() {
        let num_vars = 16;
        let n = 1 << num_vars;

        let mut rng = test_rng();
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        // Run the reference sumcheck
        let mut ref_evals = evals_ff.clone();
        let mut ref_rng = test_rng();
        let mut ref_transcript = SanityTranscript::new(&mut ref_rng);
        let ref_result = multilinear_sumcheck::<F64, F64>(&mut ref_evals, &mut ref_transcript);

        // Run the SIMD sumcheck with the same challenges
        let ref_challenges = ref_result.verifier_messages.clone();
        let mut challenge_idx = 0;

        let simd_result = prove_base_eq_ext::<MontGoldilocksNeon>(&evals_raw, |_s0, _s1| {
            let c = to_mont(ref_challenges[challenge_idx]);
            challenge_idx += 1;
            c
        });

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
            assert_eq!(to_mont(ref_msg.0), simd_msg.0, "s0 mismatch at round {}", i);
            assert_eq!(to_mont(ref_msg.1), simd_msg.1, "s1 mismatch at round {}", i);
        }
    }

    #[test]
    fn test_simd_sumcheck_small() {
        // Use actual field elements converted to Montgomery form
        let f1 = F64::from(1u64);
        let f2 = F64::from(2u64);
        let f3 = F64::from(3u64);
        let f4 = F64::from(4u64);
        let evals_raw: Vec<u64> = vec![to_mont(f1), to_mont(f2), to_mont(f3), to_mont(f4)];

        let simd_result = prove_base_eq_ext::<MontGoldilocksNeon>(&evals_raw, |_s0, _s1| {
            to_mont(F64::from(7u64))
        });

        assert_eq!(simd_result.prover_messages.len(), 2);
        assert_eq!(simd_result.verifier_messages.len(), 1);

        // Round 0: s0 = f(0)+f(2) = 1+3 = 4, s1 = f(1)+f(3) = 2+4 = 6
        assert_eq!(simd_result.prover_messages[0].0, to_mont(F64::from(4u64)));
        assert_eq!(simd_result.prover_messages[0].1, to_mont(F64::from(6u64)));

        // Round 1: after reduce with challenge=7:
        //   pair (1,2): 1 + 7*(2-1) = 8
        //   pair (3,4): 3 + 7*(4-3) = 10
        //   s0 = 8, s1 = 10
        assert_eq!(simd_result.prover_messages[1].0, to_mont(F64::from(8u64)));
        assert_eq!(simd_result.prover_messages[1].1, to_mont(F64::from(10u64)));
    }
}
