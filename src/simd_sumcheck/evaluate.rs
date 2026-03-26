//! SIMD-vectorized pairwise evaluation: computes (sum_even, sum_odd).
//!
//! Uses a 4-accumulator unroll for instruction-level parallelism.

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
/// In production, the caller should pad to this alignment.
pub fn evaluate<F: SimdBaseField>(src: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    let lanes = F::LANES;
    // Interleaved layout: even indices go to even_acc, odd indices to odd_acc.
    // With LANES=2 (Goldilocks NEON), a single load of 2 elements gives
    // one even and one odd. But the pairwise layout puts elements contiguously,
    // so we need to load 2*LANES elements and split even/odd.
    //
    // Instead, we use the simpler approach: load LANES-wide vectors and
    // accumulate. The first load is "even", the second is "odd", repeating.
    //
    // With 4-way unroll: we process 4*LANES scalars per iteration.
    // Each iteration: 4 loads, 4 adds.

    let step = 4 * lanes;
    assert!(
        src.len() % step == 0 || src.is_empty(),
        "src.len() ({}) must be a multiple of {} (4 * LANES)",
        src.len(),
        step
    );

    let zero = F::splat(F::ZERO);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

    let ptr = src.as_ptr();
    let mut i = 0;

    while i < src.len() {
        unsafe {
            acc0 = F::add(acc0, F::load(ptr.add(i)));
            acc1 = F::add(acc1, F::load(ptr.add(i + lanes)));
            acc2 = F::add(acc2, F::load(ptr.add(i + 2 * lanes)));
            acc3 = F::add(acc3, F::load(ptr.add(i + 3 * lanes)));
        }
        i += step;
    }

    // Combine accumulators: acc0, acc2 are "even groups", acc1, acc3 are "odd groups".
    // Wait — that's not right. The layout is contiguous:
    //   [0..LANES) [LANES..2*LANES) [2*LANES..3*LANES) [3*LANES..4*LANES)
    //
    // With pairwise storage [f(0), f(1), f(2), f(3), ...], and LANES=2:
    //   acc0 = [f(0)+f(4)+..., f(1)+f(5)+...]
    //   acc1 = [f(2)+f(6)+..., f(3)+f(7)+...]
    //   etc.
    //
    // So all accumulators mix even and odd. We need to reduce them lane-by-lane.
    // Combine: total = acc0 + acc1 + acc2 + acc3 (element-wise)
    let total = F::add(F::add(acc0, acc1), F::add(acc2, acc3));

    // Now `total` has LANES values. For pairwise semantics with the interleaved
    // storage [f(0), f(1), f(2), f(3), ...], each pair of adjacent elements
    // contributes:
    //   lane 0: sum of f(0), f(2), f(4), ... (even-indexed in each LANES-group)
    //   lane 1: sum of f(1), f(3), f(5), ... (odd-indexed in each LANES-group)
    //
    // Hmm, this only works cleanly if LANES=2. For LANES>2 (AVX), we need
    // a different decomposition. Let me think about this more carefully.
    //
    // Actually, the pairwise evaluation sums even-indexed and odd-indexed elements
    // from the ORIGINAL array. With LANES=2:
    //   Load [f(0), f(1)] → lane 0 is even, lane 1 is odd
    //   Load [f(2), f(3)] → lane 0 is even, lane 1 is odd
    //
    // So after accumulating, total[0] = sum of all even-indexed, total[1] = sum of all odd-indexed.
    // This is exactly what we want!
    //
    // For LANES=4 (AVX2 with u64):
    //   Load [f(0), f(1), f(2), f(3)] → lanes 0,2 are even, lanes 1,3 are odd
    //
    // So for general LANES: even lanes (0, 2, 4, ...) sum to even_total,
    // odd lanes (1, 3, 5, ...) sum to odd_total.

    // Extract lanes and sum them appropriately.
    // Store total to a temporary array, then sum even/odd lanes scalar-wise.
    let mut lanes_buf: Vec<F::Scalar> = vec![F::ZERO; F::LANES];
    unsafe { F::store(lanes_buf.as_mut_ptr(), total) };

    let mut even_sum = F::ZERO;
    let mut odd_sum = F::ZERO;
    for j in 0..F::LANES {
        if j % 2 == 0 {
            even_sum = F::scalar_add(even_sum, lanes_buf[j]);
        } else {
            odd_sum = F::scalar_add(odd_sum, lanes_buf[j]);
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

    let chunk_size = 32_768; // number of scalars per chunk
    let lanes = F::LANES;
    let step = 4 * lanes;

    // Round chunk size up to multiple of step
    let chunk_size = ((chunk_size + step - 1) / step) * step;

    // For small inputs, use the aligned+tail scalar approach directly
    if src.len() <= chunk_size {
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
        return (even, odd);
    }

    src.par_chunks(chunk_size)
        .map(|chunk| {
            // Handle last chunk that may not be aligned
            let aligned_len = (chunk.len() / step) * step;
            if aligned_len == 0 {
                // Scalar fallback for tiny remainder
                let mut even = F::ZERO;
                let mut odd = F::ZERO;
                for i in 0..chunk.len() {
                    if i % 2 == 0 {
                        even = F::scalar_add(even, chunk[i]);
                    } else {
                        odd = F::scalar_add(odd, chunk[i]);
                    }
                }
                (even, odd)
            } else {
                let (e, o) = evaluate::<F>(&chunk[..aligned_len]);
                // Handle remainder scalarly
                let mut even = e;
                let mut odd = o;
                for i in aligned_len..chunk.len() {
                    if i % 2 == 0 {
                        even = F::scalar_add(even, chunk[i]);
                    } else {
                        odd = F::scalar_add(odd, chunk[i]);
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
    let step = 4 * lanes;
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
mod tests {
    use super::*;
    use crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    use crate::tests::F64;
    use ark_ff::{PrimeField, UniformRand};
    use ark_std::test_rng;

    fn to_raw(f: F64) -> u64 {
        f.into_bigint().0[0]
    }

    #[test]
    fn test_evaluate_matches_pairwise() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        // Length must be multiple of 4*LANES = 8 for non-parallel evaluate
        let n = 1 << 16; // 65536
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_raw(*f)).collect();

        // Reference: arkworks pairwise evaluate
        let (expected_even, expected_odd) = pairwise::evaluate(&evals_ff);

        // SIMD evaluate
        let (simd_even, simd_odd) = evaluate::<GoldilocksNeon>(&evals_raw);

        assert_eq!(to_raw(expected_even), simd_even, "even sum mismatch");
        assert_eq!(to_raw(expected_odd), simd_odd, "odd sum mismatch");
    }

    #[test]
    fn test_evaluate_parallel_matches_pairwise() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 20; // ~1M elements, enough to trigger parallel chunks
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_raw(*f)).collect();

        let (expected_even, expected_odd) = pairwise::evaluate(&evals_ff);
        let (simd_even, simd_odd) = evaluate_parallel::<GoldilocksNeon>(&evals_raw);

        assert_eq!(
            to_raw(expected_even),
            simd_even,
            "parallel even sum mismatch"
        );
        assert_eq!(to_raw(expected_odd), simd_odd, "parallel odd sum mismatch");
    }
}
