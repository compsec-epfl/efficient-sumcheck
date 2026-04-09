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
    reduce_into::<F>(src, &mut out, challenge);
    out
}

/// Core SIMD reduce: reads pairs from `src` and writes folded results to `out`.
///
/// `src` must have `2 * out.len()` elements. Each pair `(src[2i], src[2i+1])`
/// produces `out[i] = src[2i] + challenge * (src[2i+1] - src[2i])`.
fn reduce_into<F: SimdBaseField>(src: &[F::Scalar], out: &mut [F::Scalar], challenge: F::Scalar) {
    let n = out.len();
    debug_assert_eq!(src.len(), 2 * n);

    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);
    let step = 4 * lanes; // 4× unroll
    let aligned = (n / step) * step;

    let src_ptr = src.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            for g in 0..4 {
                let (av, bv) = F::load_deinterleaved(src_ptr.add(2 * (i + g * lanes)));
                let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
                F::store(out_ptr.add(i + g * lanes), r);
            }
        }
        i += step;
    }

    // Handle remaining full SIMD vectors
    while i + lanes <= n {
        unsafe {
            let (av, bv) = F::load_deinterleaved(src_ptr.add(2 * i));
            let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
            F::store(out_ptr.add(i), r);
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
}

/// SIMD-vectorized pairwise reduce, in-place.
///
/// Reads pairs from the first `2*n` positions, writes results to `src[0..n]`.
/// Returns the output length `n`.
pub fn reduce_in_place<F: SimdBaseField>(src: &mut [F::Scalar], challenge: F::Scalar) -> usize {
    let n = src.len() / 2;
    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);
    let step = 4 * lanes; // 4× unroll: 4 groups of LANES outputs per iteration
    let aligned = (n / step) * step;

    let src_ptr = src.as_ptr();
    let out_ptr = src.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            for g in 0..4 {
                let (av, bv) = F::load_deinterleaved(src_ptr.add(2 * (i + g * lanes)));
                let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
                F::store(out_ptr.add(i + g * lanes), r);
            }
        }
        i += step;
    }

    while i + lanes <= n {
        unsafe {
            let (av, bv) = F::load_deinterleaved(src_ptr.add(2 * i));
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

/// Fused reduce + evaluate for the next round.
///
/// Performs in-place pairwise reduce (same as `reduce_in_place`) and simultaneously
/// accumulates the even/odd sums that `evaluate` would compute on the reduced output.
/// This eliminates one full data pass per round (the separate evaluate read).
///
/// Returns `(next_even_sum, next_odd_sum, output_length)`.
pub fn reduce_and_evaluate<F: SimdBaseField>(
    src: &mut [F::Scalar],
    challenge: F::Scalar,
) -> (F::Scalar, F::Scalar, usize) {
    let n = src.len() / 2;
    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);

    // We need 2 groups of accumulators: one for reduced values at even output
    // positions and one for odd. Within a contiguous vector of LANES elements
    // written at output position i, lanes 0,2,4,6 are "even" and 1,3,5,7 are
    // "odd" when considered as part of the flat output array (since i is always
    // aligned to LANES). So we just accumulate all reduced vectors and separate
    // even/odd lanes at the end — exactly like evaluate does.
    //
    // Use lazy accumulation: wrapping add + carry count, finalize at the end.
    // This halves the accumulation overhead (3 instructions vs 6 for full mod add).
    let zero = F::splat(F::ZERO);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;
    let mut carry0 = zero;
    let mut carry1 = zero;
    let mut carry2 = zero;
    let mut carry3 = zero;

    let step = 4 * lanes;
    let aligned = (n / step) * step;

    let src_ptr = src.as_ptr();
    let out_ptr = src.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            let (av0, bv0) = F::load_deinterleaved(src_ptr.add(2 * i));
            let r0 = F::add(av0, F::mul(challenge_v, F::sub(bv0, av0)));
            F::store(out_ptr.add(i), r0);
            let sum0 = F::add_wrapping(acc0, r0);
            carry0 = F::add_wrapping(carry0, F::carry_mask(sum0, acc0));
            acc0 = sum0;

            let (av1, bv1) = F::load_deinterleaved(src_ptr.add(2 * (i + lanes)));
            let r1 = F::add(av1, F::mul(challenge_v, F::sub(bv1, av1)));
            F::store(out_ptr.add(i + lanes), r1);
            let sum1 = F::add_wrapping(acc1, r1);
            carry1 = F::add_wrapping(carry1, F::carry_mask(sum1, acc1));
            acc1 = sum1;

            let (av2, bv2) = F::load_deinterleaved(src_ptr.add(2 * (i + 2 * lanes)));
            let r2 = F::add(av2, F::mul(challenge_v, F::sub(bv2, av2)));
            F::store(out_ptr.add(i + 2 * lanes), r2);
            let sum2 = F::add_wrapping(acc2, r2);
            carry2 = F::add_wrapping(carry2, F::carry_mask(sum2, acc2));
            acc2 = sum2;

            let (av3, bv3) = F::load_deinterleaved(src_ptr.add(2 * (i + 3 * lanes)));
            let r3 = F::add(av3, F::mul(challenge_v, F::sub(bv3, av3)));
            F::store(out_ptr.add(i + 3 * lanes), r3);
            let sum3 = F::add_wrapping(acc3, r3);
            carry3 = F::add_wrapping(carry3, F::carry_mask(sum3, acc3));
            acc3 = sum3;
        }
        i += step;
    }

    // Cleanup: single vector at a time (use full modular add — few iterations)
    while i + lanes <= n {
        unsafe {
            let (av, bv) = F::load_deinterleaved(src_ptr.add(2 * i));
            let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
            F::store(src[i..].as_mut_ptr(), r);
            acc0 = F::add(acc0, r);
        }
        i += lanes;
    }

    // Finalize lazy accumulators: correct for carries
    let red0 = F::reduce_carry(acc0, carry0);
    let red1 = F::reduce_carry(acc1, carry1);
    let red2 = F::reduce_carry(acc2, carry2);
    let red3 = F::reduce_carry(acc3, carry3);

    // Combine in a tree for ILP
    let total = F::add(F::add(red0, red1), F::add(red2, red3));

    // Extract lanes and sum even/odd groups
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

    // Scalar tail (both reduce and accumulate)
    while i < n {
        let a = src[2 * i];
        let b = src[2 * i + 1];
        let diff = F::scalar_sub(b, a);
        let scaled = F::scalar_mul(challenge, diff);
        let r = F::scalar_add(a, scaled);
        src[i] = r;
        if i % 2 == 0 {
            even_sum = F::scalar_add(even_sum, r);
        } else {
            odd_sum = F::scalar_add(odd_sum, r);
        }
        i += 1;
    }

    (even_sum, odd_sum, n)
}

/// Core fused reduce+evaluate on a src→out pair (not in-place).
///
/// Returns `(even_sum, odd_sum)` for the chunk.
fn reduce_and_evaluate_into<F: SimdBaseField>(
    src: &[F::Scalar],
    out: &mut [F::Scalar],
    challenge: F::Scalar,
) -> (F::Scalar, F::Scalar) {
    let n = out.len();
    debug_assert_eq!(src.len(), 2 * n);

    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);
    let zero = F::splat(F::ZERO);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;
    let mut carry0 = zero;
    let mut carry1 = zero;
    let mut carry2 = zero;
    let mut carry3 = zero;

    let step = 4 * lanes;
    let aligned = (n / step) * step;

    let src_ptr = src.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            let (av0, bv0) = F::load_deinterleaved(src_ptr.add(2 * i));
            let r0 = F::add(av0, F::mul(challenge_v, F::sub(bv0, av0)));
            F::store(out_ptr.add(i), r0);
            let sum0 = F::add_wrapping(acc0, r0);
            carry0 = F::add_wrapping(carry0, F::carry_mask(sum0, acc0));
            acc0 = sum0;

            let (av1, bv1) = F::load_deinterleaved(src_ptr.add(2 * (i + lanes)));
            let r1 = F::add(av1, F::mul(challenge_v, F::sub(bv1, av1)));
            F::store(out_ptr.add(i + lanes), r1);
            let sum1 = F::add_wrapping(acc1, r1);
            carry1 = F::add_wrapping(carry1, F::carry_mask(sum1, acc1));
            acc1 = sum1;

            let (av2, bv2) = F::load_deinterleaved(src_ptr.add(2 * (i + 2 * lanes)));
            let r2 = F::add(av2, F::mul(challenge_v, F::sub(bv2, av2)));
            F::store(out_ptr.add(i + 2 * lanes), r2);
            let sum2 = F::add_wrapping(acc2, r2);
            carry2 = F::add_wrapping(carry2, F::carry_mask(sum2, acc2));
            acc2 = sum2;

            let (av3, bv3) = F::load_deinterleaved(src_ptr.add(2 * (i + 3 * lanes)));
            let r3 = F::add(av3, F::mul(challenge_v, F::sub(bv3, av3)));
            F::store(out_ptr.add(i + 3 * lanes), r3);
            let sum3 = F::add_wrapping(acc3, r3);
            carry3 = F::add_wrapping(carry3, F::carry_mask(sum3, acc3));
            acc3 = sum3;
        }
        i += step;
    }

    while i + lanes <= n {
        unsafe {
            let (av, bv) = F::load_deinterleaved(src_ptr.add(2 * i));
            let r = F::add(av, F::mul(challenge_v, F::sub(bv, av)));
            F::store(out_ptr.add(i), r);
            acc0 = F::add(acc0, r);
        }
        i += lanes;
    }

    let red0 = F::reduce_carry(acc0, carry0);
    let red1 = F::reduce_carry(acc1, carry1);
    let red2 = F::reduce_carry(acc2, carry2);
    let red3 = F::reduce_carry(acc3, carry3);
    let total = F::add(F::add(red0, red1), F::add(red2, red3));

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

    while i < n {
        let a = src[2 * i];
        let b = src[2 * i + 1];
        let diff = F::scalar_sub(b, a);
        let scaled = F::scalar_mul(challenge, diff);
        let r = F::scalar_add(a, scaled);
        out[i] = r;
        if i % 2 == 0 {
            even_sum = F::scalar_add(even_sum, r);
        } else {
            odd_sum = F::scalar_add(odd_sum, r);
        }
        i += 1;
    }

    (even_sum, odd_sum)
}

/// Parallel fused reduce + evaluate using rayon.
///
/// Allocates a new output buffer, processes chunks in parallel, and returns
/// `(even_sum, odd_sum, output_vec)`.
#[cfg(feature = "parallel")]
pub fn reduce_and_evaluate_parallel<F: SimdBaseField>(
    src: &[F::Scalar],
    challenge: F::Scalar,
) -> (F::Scalar, F::Scalar, Vec<F::Scalar>) {
    use rayon::prelude::*;

    let n = src.len() / 2;
    let chunk_size = 32_768_usize;

    if n <= chunk_size {
        let mut out = vec![F::ZERO; n];
        let (e, o) = reduce_and_evaluate_into::<F>(src, &mut out, challenge);
        return (e, o, out);
    }

    let mut out = vec![F::ZERO; n];
    let pair_chunk = chunk_size * 2;

    let (even, odd) = out
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(idx, out_chunk)| {
            let src_start = idx * pair_chunk;
            let src_end = (src_start + out_chunk.len() * 2).min(src.len());
            reduce_and_evaluate_into::<F>(&src[src_start..src_end], out_chunk, challenge)
        })
        .reduce(
            || (F::ZERO, F::ZERO),
            |(e1, o1), (e2, o2)| (F::scalar_add(e1, e2), F::scalar_add(o1, o2)),
        );

    (even, odd, out)
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
pub fn reduce_and_evaluate_parallel<F: SimdBaseField>(
    src: &[F::Scalar],
    challenge: F::Scalar,
) -> (F::Scalar, F::Scalar, Vec<F::Scalar>) {
    let n = src.len() / 2;
    let mut out = vec![F::ZERO; n];
    let (e, o) = reduce_and_evaluate_into::<F>(src, &mut out, challenge);
    (e, o, out)
}

/// Parallel SIMD reduce (producing a new Vec).
///
/// Pre-allocates the output and writes directly to non-overlapping slices
/// via `par_chunks_mut`, avoiding per-chunk Vec allocations.
#[cfg(feature = "parallel")]
pub fn reduce_parallel<F: SimdBaseField>(
    src: &[F::Scalar],
    challenge: F::Scalar,
) -> Vec<F::Scalar> {
    use rayon::prelude::*;

    let n = src.len() / 2;
    let chunk_size = 32_768_usize;

    if n <= chunk_size {
        return reduce_to_vec::<F>(src, challenge);
    }

    let mut out = vec![F::ZERO; n];
    let pair_chunk = chunk_size * 2;

    out.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(idx, out_chunk)| {
            let src_start = idx * pair_chunk;
            let src_end = src_start + out_chunk.len() * 2;
            reduce_into::<F>(&src[src_start..src_end], out_chunk, challenge);
        });

    out
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

        let received_raw = reduce_to_vec::<Backend>(&evals_raw, challenge_raw);

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
    fn test_reduce_and_evaluate_matches() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 16;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        let challenge_ff = F64::rand(&mut rng);
        let challenge_raw = to_mont(challenge_ff);

        // Reference: reduce then evaluate
        let mut expected_ff = evals_ff;
        pairwise::reduce_evaluations(&mut expected_ff, challenge_ff);
        let (expected_even, expected_odd) = pairwise::evaluate(&expected_ff);

        // Fused
        let (fused_even, fused_odd, new_len) =
            reduce_and_evaluate::<Backend>(&mut evals_raw, challenge_raw);

        assert_eq!(new_len, n / 2);
        assert_eq!(to_mont(expected_even), fused_even, "fused even mismatch");
        assert_eq!(to_mont(expected_odd), fused_odd, "fused odd mismatch");

        // Also verify the reduce output matches
        for i in 0..new_len {
            assert_eq!(
                to_mont(expected_ff[i]),
                evals_raw[i],
                "reduce mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_reduce_and_evaluate_large() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 20;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        let challenge_ff = F64::rand(&mut rng);
        let challenge_raw = to_mont(challenge_ff);

        let mut expected_ff = evals_ff;
        pairwise::reduce_evaluations(&mut expected_ff, challenge_ff);
        let (expected_even, expected_odd) = pairwise::evaluate(&expected_ff);

        let (fused_even, fused_odd, _) =
            reduce_and_evaluate::<Backend>(&mut evals_raw, challenge_raw);

        assert_eq!(
            to_mont(expected_even),
            fused_even,
            "large fused even mismatch"
        );
        assert_eq!(to_mont(expected_odd), fused_odd, "large fused odd mismatch");
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

        let received_raw = reduce_parallel::<Backend>(&evals_raw, challenge_raw);

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
