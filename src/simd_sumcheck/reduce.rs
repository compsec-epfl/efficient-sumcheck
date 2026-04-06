//! SIMD-vectorized pairwise reduce: folds evaluations with a challenge.
//!
//! For each adjacent pair `(a, b)`: `result = a + challenge * (b - a)`
//!
//! This is the base-field reduce used when base = extension (EXT_DEGREE = 1).

use crate::simd_fields::SimdBaseField;

/// SIMD-vectorized pairwise reduce, producing a new Vec.
pub fn reduce_to_vec<F: SimdBaseField>(src: &[F::Scalar], challenge: F::Scalar) -> Vec<F::Scalar> {
    let n = src.len() / 2;
    let mut out = vec![F::ZERO; n];

    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);
    let aligned = (n / lanes) * lanes;

    for i in (0..aligned).step_by(lanes) {
        let src_idx = 2 * i;
        let mut a_buf = vec![F::ZERO; lanes];
        let mut b_buf = vec![F::ZERO; lanes];

        for j in 0..lanes {
            a_buf[j] = src[src_idx + 2 * j];
            b_buf[j] = src[src_idx + 2 * j + 1];
        }

        unsafe {
            let a_v = F::load(a_buf.as_ptr());
            let b_v = F::load(b_buf.as_ptr());
            let diff = F::sub(b_v, a_v);
            let scaled = F::mul(challenge_v, diff);
            let result = F::add(a_v, scaled);
            F::store(out[i..].as_mut_ptr(), result);
        }
    }

    // Scalar tail
    for i in aligned..n {
        let a = src[2 * i];
        let b = src[2 * i + 1];
        let diff = F::scalar_sub(b, a);
        let scaled = F::scalar_mul(challenge, diff);
        out[i] = F::scalar_add(a, scaled);
    }

    out
}

/// Parallel SIMD reduce (producing a new Vec).
#[cfg(feature = "parallel")]
pub fn reduce_parallel<F: SimdBaseField>(
    src: &[F::Scalar],
    challenge: F::Scalar,
) -> Vec<F::Scalar> {
    use rayon::prelude::*;

    let n = src.len() / 2;
    let chunk_size = 32_768_usize;
    let pair_chunk = chunk_size * 2;

    if n <= chunk_size {
        return reduce_to_vec::<F>(src, challenge);
    }

    src.par_chunks(pair_chunk)
        .flat_map(|chunk| reduce_to_vec::<F>(chunk, challenge))
        .collect()
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
pub fn reduce_parallel<F: SimdBaseField>(
    src: &[F::Scalar],
    challenge: F::Scalar,
) -> Vec<F::Scalar> {
    reduce_to_vec::<F>(src, challenge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    use crate::tests::{to_mont, F64};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_reduce_matches_pairwise() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 16;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        let challenge_ff = F64::rand(&mut rng);
        let challenge_raw = to_mont(challenge_ff);

        let mut expected_ff = evals_ff.clone();
        pairwise::reduce_evaluations(&mut expected_ff, challenge_ff);

        let received_raw = reduce_to_vec::<GoldilocksNeon>(&evals_raw, challenge_raw);

        assert_eq!(expected_ff.len(), received_raw.len());
        for i in 0..expected_ff.len() {
            assert_eq!(
                to_mont(expected_ff[i]),
                received_raw[i],
                "mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_reduce_parallel_matches() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 20;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        let challenge_ff = F64::rand(&mut rng);
        let challenge_raw = to_mont(challenge_ff);

        let mut expected_ff = evals_ff;
        pairwise::reduce_evaluations(&mut expected_ff, challenge_ff);

        let received_raw = reduce_parallel::<GoldilocksNeon>(&evals_raw, challenge_raw);

        assert_eq!(expected_ff.len(), received_raw.len());
        for i in 0..expected_ff.len() {
            assert_eq!(
                to_mont(expected_ff[i]),
                received_raw[i],
                "mismatch at index {}",
                i
            );
        }
    }
}
