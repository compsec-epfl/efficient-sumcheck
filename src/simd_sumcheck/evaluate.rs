//! SIMD-vectorized pairwise evaluation: computes (sum_even, sum_odd).
//!
//! Uses an 8-accumulator unroll for instruction-level parallelism,
//! which is the sweet spot on NEON (saturates the register file without
//! spilling — see "Proof Systems Engineering" for benchmarking methodology).

use crate::simd_fields::SimdBaseField;

/// SIMD-vectorized pairwise evaluate.
///
/// Given `src` = `[f(0), f(1), f(2), f(3), ...]`, computes:
///   sum_even = f(0) + f(2) + f(4) + ...
///   sum_odd  = f(1) + f(3) + f(5) + ...
///
/// Returns `(sum_even, sum_odd)`.
///
/// # Panics
///
/// Panics if `src.len()` is not a multiple of `8 * F::LANES` (the unroll factor).
pub fn evaluate<F: SimdBaseField>(src: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    let lanes = F::LANES;
    let step = 8 * lanes;
    assert!(
        src.len() % step == 0 || src.is_empty(),
        "src.len() ({}) must be a multiple of {} (8 * LANES)",
        src.len(),
        step
    );

    let zero = F::splat(F::ZERO);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;
    let mut acc4 = zero;
    let mut acc5 = zero;
    let mut acc6 = zero;
    let mut acc7 = zero;

    let ptr = src.as_ptr();
    let mut i = 0;

    while i < src.len() {
        unsafe {
            acc0 = F::add(acc0, F::load(ptr.add(i)));
            acc1 = F::add(acc1, F::load(ptr.add(i + lanes)));
            acc2 = F::add(acc2, F::load(ptr.add(i + 2 * lanes)));
            acc3 = F::add(acc3, F::load(ptr.add(i + 3 * lanes)));
            acc4 = F::add(acc4, F::load(ptr.add(i + 4 * lanes)));
            acc5 = F::add(acc5, F::load(ptr.add(i + 5 * lanes)));
            acc6 = F::add(acc6, F::load(ptr.add(i + 6 * lanes)));
            acc7 = F::add(acc7, F::load(ptr.add(i + 7 * lanes)));
        }
        i += step;
    }

    // Combine accumulators in a tree to keep ILP.
    let total = F::add(
        F::add(F::add(acc0, acc1), F::add(acc2, acc3)),
        F::add(F::add(acc4, acc5), F::add(acc6, acc7)),
    );

    // Extract lanes and sum even/odd groups.
    let mut lanes_buf = [F::ZERO; 16];
    debug_assert!(F::LANES <= 16);
    unsafe { F::store(lanes_buf.as_mut_ptr(), total) };

    let mut even_sum = F::ZERO;
    let mut odd_sum = F::ZERO;
    for (j, &val) in lanes_buf.iter().enumerate().take(F::LANES) {
        if j % 2 == 0 {
            even_sum = F::scalar_add(even_sum, val);
        } else {
            odd_sum = F::scalar_add(odd_sum, val);
        }
    }

    (even_sum, odd_sum)
}

/// Parallel SIMD evaluate with chunking for large arrays.
///
/// Splits `src` into chunks, evaluates each in parallel (when the `parallel`
/// feature is enabled), then combines.
#[cfg(feature = "parallel")]
pub fn evaluate_parallel<F: SimdBaseField>(src: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    use rayon::prelude::*;

    let chunk_size: usize = 32_768;
    let lanes = F::LANES;
    let step = 8 * lanes;
    let chunk_size = chunk_size.div_ceil(step) * step;

    if src.len() <= chunk_size {
        let aligned_len = (src.len() / step) * step;
        let (mut even, mut odd) = if aligned_len > 0 {
            evaluate::<F>(&src[..aligned_len])
        } else {
            (F::ZERO, F::ZERO)
        };
        for (i, &val) in src.iter().enumerate().skip(aligned_len) {
            if i % 2 == 0 {
                even = F::scalar_add(even, val);
            } else {
                odd = F::scalar_add(odd, val);
            }
        }
        return (even, odd);
    }

    src.par_chunks(chunk_size)
        .map(|chunk| {
            let aligned_len = (chunk.len() / step) * step;
            if aligned_len == 0 {
                let mut even = F::ZERO;
                let mut odd = F::ZERO;
                for (i, &val) in chunk.iter().enumerate() {
                    if i % 2 == 0 {
                        even = F::scalar_add(even, val);
                    } else {
                        odd = F::scalar_add(odd, val);
                    }
                }
                (even, odd)
            } else {
                let (e, o) = evaluate::<F>(&chunk[..aligned_len]);
                let mut even = e;
                let mut odd = o;
                for (i, &val) in chunk.iter().enumerate().skip(aligned_len) {
                    if i % 2 == 0 {
                        even = F::scalar_add(even, val);
                    } else {
                        odd = F::scalar_add(odd, val);
                    }
                }
                (even, odd)
            }
        })
        .reduce(
            || (F::ZERO, F::ZERO),
            |(e1, o1), (e2, o2)| (F::scalar_add(e1, e2), F::scalar_add(o1, o2)),
        )
}

/// Non-parallel version of evaluate that handles arbitrary lengths.
#[cfg(not(feature = "parallel"))]
pub fn evaluate_parallel<F: SimdBaseField>(src: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    let lanes = F::LANES;
    let step = 8 * lanes;
    let aligned_len = (src.len() / step) * step;

    let (mut even, mut odd) = if aligned_len > 0 {
        evaluate::<F>(&src[..aligned_len])
    } else {
        (F::ZERO, F::ZERO)
    };

    for i in aligned_len..src.len() {
        if i % 2 == 0 {
            even = F::scalar_add(even, src[i]);
        } else {
            odd = F::scalar_add(odd, src[i]);
        }
    }

    (even, odd)
}

#[cfg(test)]
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
mod tests {
    use super::*;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    use crate::simd_fields::goldilocks::avx512::GoldilocksAvx512 as Backend;
    #[cfg(target_arch = "aarch64")]
    use crate::simd_fields::goldilocks::neon::GoldilocksNeon as Backend;
    use crate::tests::{to_mont, F64};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_evaluate_matches_pairwise() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 16;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        // Reference: arkworks pairwise evaluate
        let (expected_even, expected_odd) = pairwise::evaluate(&evals_ff);

        // SIMD evaluate (Montgomery domain)
        let (simd_even, simd_odd) = evaluate::<Backend>(&evals_raw);

        assert_eq!(to_mont(expected_even), simd_even, "even sum mismatch");
        assert_eq!(to_mont(expected_odd), simd_odd, "odd sum mismatch");
    }

    #[test]
    fn test_evaluate_parallel_matches_pairwise() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 20;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        let (expected_even, expected_odd) = pairwise::evaluate(&evals_ff);
        let (simd_even, simd_odd) = evaluate_parallel::<Backend>(&evals_raw);

        assert_eq!(
            to_mont(expected_even),
            simd_even,
            "parallel even sum mismatch"
        );
        assert_eq!(to_mont(expected_odd), simd_odd, "parallel odd sum mismatch");
    }
}
