//! SIMD-vectorized pairwise reduce: folds evaluations with a challenge.
//!
//! For each adjacent pair `(a, b)`: `result = a + challenge * (b - a)`
//!
//! This is the base-field reduce used when base = extension (EXT_DEGREE = 1).

use crate::simd_fields::SimdBaseField;

/// SIMD-vectorized pairwise reduce, producing a new Vec.
///
/// Uses 4× loop unrolling for instruction-level parallelism.
/// (8× was benchmarked but regressed due to register pressure from mul.)
/// Stack-allocated deinterleave buffers avoid per-iteration heap allocation.
pub fn reduce_to_vec<F: SimdBaseField>(src: &[F::Scalar], challenge: F::Scalar) -> Vec<F::Scalar> {
    let n = src.len() / 2;
    let mut out = vec![F::ZERO; n];

    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);
    let step = 4 * lanes; // 4× unroll
    let aligned = (n / step) * step;

    // Stack-allocated deinterleave buffers (LANES is small: 2 for NEON u64).
    debug_assert!(lanes <= 16);
    let mut ab = [([F::ZERO; 16], [F::ZERO; 16]); 4];

    let mut i = 0;
    while i < aligned {
        // Deinterleave 4 groups of LANES pairs
        for g in 0..4 {
            for j in 0..lanes {
                let s = 2 * (i + g * lanes + j);
                ab[g].0[j] = src[s];
                ab[g].1[j] = src[s + 1];
            }
        }

        unsafe {
            for g in 0..4 {
                let av = F::load(ab[g].0.as_ptr());
                let bv = F::load(ab[g].1.as_ptr());
                let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
                F::store(out[i + g * lanes..].as_mut_ptr(), r);
            }
        }

        i += step;
    }

    // Handle remaining full SIMD vectors (1–3 vectors that didn't fill a 4× group)
    while i + lanes <= n {
        for j in 0..lanes {
            ab[0].0[j] = src[2 * (i + j)];
            ab[0].1[j] = src[2 * (i + j) + 1];
        }
        unsafe {
            let av = F::load(ab[0].0.as_ptr());
            let bv = F::load(ab[0].1.as_ptr());
            let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
            F::store(out[i..].as_mut_ptr(), r);
        }
        i += lanes;
    }

    // Scalar tail
    while i < n {
        let a = src[2 * i];
        let b = src[2 * i + 1];
        let diff = F::scalar_sub(b, a);
        let scaled = F::scalar_mul(challenge, diff);
        out[i] = F::scalar_add(a, scaled);
        i += 1;
    }

    out
}

/// SIMD-vectorized pairwise reduce, in-place.
///
/// Reads pairs from the first `2*n` positions, writes results to `src[0..n]`.
/// Returns the output length `n`.
pub fn reduce_in_place<F: SimdBaseField>(src: &mut [F::Scalar], challenge: F::Scalar) -> usize {
    let n = src.len() / 2;
    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);
    let step = 4 * lanes;
    let aligned = (n / step) * step;

    debug_assert!(lanes <= 16);
    let mut ab = [([F::ZERO; 16], [F::ZERO; 16]); 4];

    let mut i = 0;
    while i < aligned {
        for g in 0..4 {
            for j in 0..lanes {
                let s = 2 * (i + g * lanes + j);
                ab[g].0[j] = src[s];
                ab[g].1[j] = src[s + 1];
            }
        }

        unsafe {
            for g in 0..4 {
                let av = F::load(ab[g].0.as_ptr());
                let bv = F::load(ab[g].1.as_ptr());
                let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
                F::store(src[i + g * lanes..].as_mut_ptr(), r);
            }
        }

        i += step;
    }

    while i + lanes <= n {
        for j in 0..lanes {
            ab[0].0[j] = src[2 * (i + j)];
            ab[0].1[j] = src[2 * (i + j) + 1];
        }
        unsafe {
            let av = F::load(ab[0].0.as_ptr());
            let bv = F::load(ab[0].1.as_ptr());
            let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
            F::store(src[i..].as_mut_ptr(), r);
        }
        i += lanes;
    }

    while i < n {
        let a = src[2 * i];
        let b = src[2 * i + 1];
        let diff = F::scalar_sub(b, a);
        let scaled = F::scalar_mul(challenge, diff);
        src[i] = F::scalar_add(a, scaled);
        i += 1;
    }

    n
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
