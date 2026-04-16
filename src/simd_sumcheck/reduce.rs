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

/// Reduce both f and g in-place in a single interleaved streaming pass.
///
/// Instead of two separate `reduce_in_place` calls (2 full data passes),
/// this reads f and g pairs together, saving cache/bandwidth.
/// Returns the output length `n`.
pub fn reduce_both_in_place<F: SimdBaseField>(
    f: &mut [F::Scalar],
    g: &mut [F::Scalar],
    challenge: F::Scalar,
) -> usize {
    let n = f.len() / 2;
    debug_assert_eq!(f.len(), g.len());
    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);
    let step = 4 * lanes;
    let aligned = (n / step) * step;

    let f_ptr = f.as_ptr();
    let g_ptr = g.as_ptr();
    let f_out = f.as_mut_ptr();
    let g_out = g.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            for u in 0..4 {
                let off = i + u * lanes;

                let (fv_a, fv_b) = F::load_deinterleaved(f_ptr.add(2 * off));
                let f_red = F::add(fv_a, F::mul(challenge_v, F::sub(fv_b, fv_a)));
                F::store(f_out.add(off), f_red);

                let (gv_a, gv_b) = F::load_deinterleaved(g_ptr.add(2 * off));
                let g_red = F::add(gv_a, F::mul(challenge_v, F::sub(gv_b, gv_a)));
                F::store(g_out.add(off), g_red);
            }
        }
        i += step;
    }

    while i < n {
        let fa = f[2 * i];
        let fb = f[2 * i + 1];
        f[i] = F::scalar_add(fa, F::scalar_mul(challenge, F::scalar_sub(fb, fa)));

        let ga = g[2 * i];
        let gb = g[2 * i + 1];
        g[i] = F::scalar_add(ga, F::scalar_mul(challenge, F::scalar_sub(gb, ga)));

        i += 1;
    }

    n
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
    let mut lanes_buf = [F::ZERO; 32];
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

    let mut lanes_buf = [F::ZERO; 32];
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

// ── Extension field reduce ──────────────────────────────────────────────────

/// Degree-2 extension reduce in-place.
///
/// `src` contains `n` extension elements as `2*n` consecutive u64s in AoS layout.
/// `challenge` is the extension challenge as `[c0, c1]` raw u64s.
/// `w` is the nonresidue in Montgomery form.
///
/// For each pair `(a, b)`: `result = a + challenge * (b - a)` using ext2 multiply.
/// Returns the new length in u64s (`n * ext_degree / 2 = n`).
/// Degree-2 extension reduce, producing a new Vec (parallel-friendly).
///
/// Each pair of adjacent extension elements `(a, b)` is folded:
/// `result = a + challenge * (b - a)` using degree-2 Karatsuba.
///
/// `src` is `n_elems * 2` u64s in AoS layout. Returns `n_elems/2 * 2` u64s.
#[cfg(feature = "parallel")]
pub fn ext2_reduce_parallel(src: &[u64], challenge: [u64; 2], w: u64) -> Vec<u64> {
    use rayon::prelude::*;

    let ext_deg = 2;
    let pair_u64s = 2 * ext_deg; // 4 u64s per pair (even + odd element)
    let n_pairs = src.len() / pair_u64s;
    let chunk_pairs = 16_384_usize;
    let chunk_u64s = chunk_pairs * pair_u64s;

    if n_pairs <= chunk_pairs {
        return ext2_reduce_chunk(src, challenge, w);
    }

    src.par_chunks(chunk_u64s)
        .flat_map(|chunk| ext2_reduce_chunk(chunk, challenge, w))
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub fn ext2_reduce_parallel(src: &[u64], challenge: [u64; 2], w: u64) -> Vec<u64> {
    ext2_reduce_chunk(src, challenge, w)
}

/// Process a chunk of pairs for ext2 reduce.
///
/// Uses precomputed `c1w = c1 * w` for the "mul-by-constant matrix" approach:
/// 4 base muls + 2 adds instead of Karatsuba's 3 muls + 5 adds.
fn ext2_reduce_chunk(src: &[u64], challenge: [u64; 2], w: u64) -> Vec<u64> {
    let ext_deg = 2;
    let n_elems = src.len() / ext_deg;
    let n_pairs = n_elems / 2;
    let mut out = vec![0u64; n_pairs * ext_deg];

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_fields::goldilocks::neon::GoldilocksNeon;
        use crate::simd_fields::SimdBaseField;

        // Precompute c1*w once for this chunk (same challenge for all pairs)
        let c0 = challenge[0];
        let c1 = challenge[1];
        let c1w = GoldilocksNeon::scalar_mul(c1, w);

        for i in 0..n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            let d0 = GoldilocksNeon::scalar_sub(src[b_off], src[a_off]);
            let d1 = GoldilocksNeon::scalar_sub(src[b_off + 1], src[a_off + 1]);

            // (c0, c1) * (d0, d1) mod (X² - w) using precomputed c1w:
            //   prod0 = c0*d0 + c1w*d1
            //   prod1 = c0*d1 + c1*d0
            let prod0 = GoldilocksNeon::scalar_add(
                GoldilocksNeon::scalar_mul(c0, d0),
                GoldilocksNeon::scalar_mul(c1w, d1),
            );
            let prod1 = GoldilocksNeon::scalar_add(
                GoldilocksNeon::scalar_mul(c0, d1),
                GoldilocksNeon::scalar_mul(c1, d0),
            );

            out[out_off] = GoldilocksNeon::scalar_add(src[a_off], prod0);
            out[out_off + 1] = GoldilocksNeon::scalar_add(src[a_off + 1], prod1);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        use crate::simd_fields::goldilocks::avx512::{
            ext2_reduce_8pairs, ext2_scalar_mul, GoldilocksAvx512,
        };

        let challenge_c0 = GoldilocksAvx512::splat(challenge[0]);
        let challenge_c1 = GoldilocksAvx512::splat(challenge[1]);
        let w_vec = GoldilocksAvx512::splat(w);

        // Process 8 pairs at a time (32 input u64s → 16 output u64s)
        let simd_pairs = (n_pairs / 8) * 8;
        let mut i = 0;
        while i < simd_pairs {
            let src_off = (2 * i) * ext_deg; // 4 u64s per pair, 8 pairs = 32 u64s
            let out_off = i * ext_deg; // 2 u64s per result, 8 results = 16 u64s
            unsafe {
                ext2_reduce_8pairs(
                    src.as_ptr().add(src_off),
                    out.as_mut_ptr().add(out_off),
                    challenge_c0,
                    challenge_c1,
                    w_vec,
                );
            }
            i += 8;
        }

        // Scalar tail for remaining pairs
        while i < n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            let diff = [
                GoldilocksAvx512::scalar_sub(src[b_off], src[a_off]),
                GoldilocksAvx512::scalar_sub(src[b_off + 1], src[a_off + 1]),
            ];
            let prod = ext2_scalar_mul(diff, challenge, w);
            out[out_off] = GoldilocksAvx512::scalar_add(src[a_off], prod[0]);
            out[out_off + 1] = GoldilocksAvx512::scalar_add(src[a_off + 1], prod[1]);
            i += 1;
        }
    }

    out
}

/// Degree-2 extension reduce in-place (single-threaded, for small inputs).
/// Fused ext2 reduce + next-round evaluate.
///
/// In one pass over the data:
/// 1. Reduces each pair (a, b) → result = a + challenge * (b - a) using ext2 Karatsuba
/// 2. Accumulates even/odd sums of the reduced output (next round's evaluate)
/// 3. Stores reduced data in-place (front half of src)
///
/// Returns `(even_components, odd_components, new_length_u64)` where
/// even/odd are `[c0, c1]` raw u64 component sums.
///
/// This eliminates one full data pass per round vs separate reduce + evaluate.
#[cfg(target_arch = "aarch64")]
pub fn ext2_reduce_and_evaluate(
    src: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2], usize) {
    use crate::simd_fields::goldilocks::neon::GoldilocksNeon;

    let ext_deg = 2;
    let n_elems = src.len() / ext_deg;
    let n_pairs = n_elems / 2;
    let n_out_elems = n_pairs;

    // Precompute c1*w once
    let c0 = challenge[0];
    let c1 = challenge[1];
    let c1w = GoldilocksNeon::scalar_mul(c1, w);

    let mut even_c0: u64 = 0;
    let mut even_c1: u64 = 0;
    let mut odd_c0: u64 = 0;
    let mut odd_c1: u64 = 0;

    for i in 0..n_pairs {
        let a_off = (2 * i) * ext_deg;
        let b_off = (2 * i + 1) * ext_deg;
        let out_off = i * ext_deg;

        let a = [src[a_off], src[a_off + 1]];
        let b = [src[b_off], src[b_off + 1]];

        let d0 = GoldilocksNeon::scalar_sub(b[0], a[0]);
        let d1 = GoldilocksNeon::scalar_sub(b[1], a[1]);

        // Precomputed mul-by-constant: 4 base muls + 2 adds
        let prod0 = GoldilocksNeon::scalar_add(
            GoldilocksNeon::scalar_mul(c0, d0),
            GoldilocksNeon::scalar_mul(c1w, d1),
        );
        let prod1 = GoldilocksNeon::scalar_add(
            GoldilocksNeon::scalar_mul(c0, d1),
            GoldilocksNeon::scalar_mul(c1, d0),
        );
        let prod = [prod0, prod1];

        // result = a + product
        let r0 = GoldilocksNeon::scalar_add(a[0], prod[0]);
        let r1 = GoldilocksNeon::scalar_add(a[1], prod[1]);

        // Store reduced result
        src[out_off] = r0;
        src[out_off + 1] = r1;

        // Accumulate into even/odd based on output extension element index
        if i % 2 == 0 {
            even_c0 = GoldilocksNeon::scalar_add(even_c0, r0);
            even_c1 = GoldilocksNeon::scalar_add(even_c1, r1);
        } else {
            odd_c0 = GoldilocksNeon::scalar_add(odd_c0, r0);
            odd_c1 = GoldilocksNeon::scalar_add(odd_c1, r1);
        }
    }

    ([even_c0, even_c1], [odd_c0, odd_c1], n_out_elems * ext_deg)
}

/// Fused ext3 reduce + next-round evaluate.
///
/// Same concept as ext2 but for degree-3 extensions (6 Karatsuba base muls).
#[cfg(target_arch = "aarch64")]
pub fn ext3_reduce_and_evaluate(
    src: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3], usize) {
    use crate::simd_fields::goldilocks::neon::{ext3_scalar_mul, GoldilocksNeon};

    let ext_deg = 3;
    let n_elems = src.len() / ext_deg;
    let n_pairs = n_elems / 2;
    let n_out_elems = n_pairs;

    let mut even = [0u64; 3];
    let mut odd = [0u64; 3];

    for i in 0..n_pairs {
        let a_off = (2 * i) * ext_deg;
        let b_off = (2 * i + 1) * ext_deg;
        let out_off = i * ext_deg;

        let a = [src[a_off], src[a_off + 1], src[a_off + 2]];
        let b = [src[b_off], src[b_off + 1], src[b_off + 2]];

        let diff = [
            GoldilocksNeon::scalar_sub(b[0], a[0]),
            GoldilocksNeon::scalar_sub(b[1], a[1]),
            GoldilocksNeon::scalar_sub(b[2], a[2]),
        ];

        let prod = ext3_scalar_mul(diff, challenge, w);

        let r = [
            GoldilocksNeon::scalar_add(a[0], prod[0]),
            GoldilocksNeon::scalar_add(a[1], prod[1]),
            GoldilocksNeon::scalar_add(a[2], prod[2]),
        ];

        src[out_off] = r[0];
        src[out_off + 1] = r[1];
        src[out_off + 2] = r[2];

        if i % 2 == 0 {
            for c in 0..3 {
                even[c] = GoldilocksNeon::scalar_add(even[c], r[c]);
            }
        } else {
            for c in 0..3 {
                odd[c] = GoldilocksNeon::scalar_add(odd[c], r[c]);
            }
        }
    }

    (even, odd, n_out_elems * ext_deg)
}

/// Fused inner-product round: evaluate (a, b) + reduce both f and g in one pass.
///
/// In a single streaming pass over f and g:
/// 1. Loads (f0,f1) and (g0,g1) pairs via deinterleaved reads
/// 2. Accumulates a += f0*g0, b += f0*g1 + f1*g0 (product evaluate)
/// 3. Stores f' = f0 + r*(f1-f0) and g' = g0 + r*(g1-g0) in front halves
///
/// Returns (a, b, new_len) where a,b are the prover message coefficients.
pub fn product_reduce_and_evaluate<F: SimdBaseField>(
    f: &mut [F::Scalar],
    g: &mut [F::Scalar],
    challenge: F::Scalar,
) -> (F::Scalar, F::Scalar, usize) {
    let n = f.len() / 2;
    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);

    let mut acc_a = F::splat(F::ZERO); // Σ f_even * g_even
    let mut acc_b = F::splat(F::ZERO); // Σ (f_even*g_odd + f_odd*g_even)

    let f_ptr = f.as_ptr();
    let g_ptr = g.as_ptr();
    let f_out = f.as_mut_ptr();
    let g_out = g.as_mut_ptr();

    let step = 4 * lanes;
    let aligned = (n / step) * step;

    let mut i = 0;
    while i < aligned {
        unsafe {
            for u in 0..4 {
                let off = i + u * lanes;
                let (fe, fo) = F::load_deinterleaved(f_ptr.add(2 * off));
                let (ge, go) = F::load_deinterleaved(g_ptr.add(2 * off));

                // Accumulate product evaluate
                acc_a = F::add(acc_a, F::mul(fe, ge));
                acc_b = F::add(acc_b, F::add(F::mul(fe, go), F::mul(fo, ge)));

                // Reduce: f' = fe + r*(fo - fe), g' = ge + r*(go - ge)
                let f_red = F::add(fe, F::mul(challenge_v, F::sub(fo, fe)));
                let g_red = F::add(ge, F::mul(challenge_v, F::sub(go, ge)));
                F::store(f_out.add(off), f_red);
                F::store(g_out.add(off), g_red);
            }
        }
        i += step;
    }

    // Horizontal sum of SIMD accumulators
    let mut buf = [F::ZERO; 32];
    let mut a_sum = F::ZERO;
    let mut b_sum = F::ZERO;
    unsafe { F::store(buf.as_mut_ptr(), acc_a) };
    for &v in buf.iter().take(lanes) {
        a_sum = F::scalar_add(a_sum, v);
    }
    unsafe { F::store(buf.as_mut_ptr(), acc_b) };
    for &v in buf.iter().take(lanes) {
        b_sum = F::scalar_add(b_sum, v);
    }

    // Scalar tail: evaluate + reduce for remaining pairs
    while i < n {
        let fe = f[2 * i];
        let fo = f[2 * i + 1];
        let ge = g[2 * i];
        let go = g[2 * i + 1];

        a_sum = F::scalar_add(a_sum, F::scalar_mul(fe, ge));
        b_sum = F::scalar_add(
            b_sum,
            F::scalar_add(F::scalar_mul(fe, go), F::scalar_mul(fo, ge)),
        );

        f[i] = F::scalar_add(fe, F::scalar_mul(challenge, F::scalar_sub(fo, fe)));
        g[i] = F::scalar_add(ge, F::scalar_mul(challenge, F::scalar_sub(go, ge)));

        i += 1;
    }

    (a_sum, b_sum, n)
}

#[allow(dead_code)]
pub fn ext2_reduce_in_place<F: SimdBaseField<Scalar = u64>>(
    src: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> usize {
    let ext_deg = 2;
    let n_elems = src.len() / ext_deg;
    let n_pairs = n_elems / 2;

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_fields::goldilocks::neon::{ext2_scalar_mul, GoldilocksNeon};

        let _w_vec = GoldilocksNeon::splat(w);
        let _chg_v = [
            GoldilocksNeon::splat(challenge[0]),
            GoldilocksNeon::splat(challenge[1]),
        ];

        // With NEON LANES=2 and degree-2: one SIMD load = one extension element.
        // Process pairs: load even (2 u64s), load odd (2 u64s), compute result.
        let ptr = src.as_mut_ptr();
        for i in 0..n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            unsafe {
                // Load even and odd extension elements
                let a_v = GoldilocksNeon::load(ptr.add(a_off) as *const u64);
                let b_v = GoldilocksNeon::load(ptr.add(b_off) as *const u64);

                // diff = b - a (component-wise, both components in one SIMD op)
                let diff_v = GoldilocksNeon::sub(b_v, a_v);

                // For ext2 multiply, we need SoA: separate c0 and c1 components.
                // With LANES=2, the vector holds [c0, c1] — need to broadcast
                // each component to both lanes for the multiply.
                // Actually, ext2_mul expects [Packed; 2] where each Packed has
                // the same component from multiple elements. With only 1 element
                // per SIMD vector, we just extract and use scalar.
                let diff0 = core::arch::aarch64::vgetq_lane_u64(diff_v, 0);
                let diff1 = core::arch::aarch64::vgetq_lane_u64(diff_v, 1);
                let prod = ext2_scalar_mul([diff0, diff1], challenge, w);

                // result = a + prod (component-wise)
                let a0 = core::arch::aarch64::vgetq_lane_u64(a_v, 0);
                let a1 = core::arch::aarch64::vgetq_lane_u64(a_v, 1);
                let r0 = GoldilocksNeon::scalar_add(a0, prod[0]);
                let r1 = GoldilocksNeon::scalar_add(a1, prod[1]);

                *ptr.add(out_off) = r0;
                *ptr.add(out_off + 1) = r1;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        use crate::simd_fields::goldilocks::avx512::{
            ext2_reduce_8pairs, ext2_scalar_mul, GoldilocksAvx512,
        };

        let challenge_c0 = GoldilocksAvx512::splat(challenge[0]);
        let challenge_c1 = GoldilocksAvx512::splat(challenge[1]);
        let w_vec = GoldilocksAvx512::splat(w);

        let ptr = src.as_mut_ptr();
        let simd_pairs = (n_pairs / 8) * 8;
        let mut i = 0;

        // Safe in-place: ext2_reduce_8pairs loads all 32 u64s into registers
        // before writing 16 u64s, and output region is always <= input region.
        while i < simd_pairs {
            let src_off = (2 * i) * ext_deg;
            let out_off = i * ext_deg;
            unsafe {
                ext2_reduce_8pairs(
                    ptr.add(src_off) as *const u64,
                    ptr.add(out_off),
                    challenge_c0,
                    challenge_c1,
                    w_vec,
                );
            }
            i += 8;
        }

        while i < n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            let diff = [
                GoldilocksAvx512::scalar_sub(src[b_off], src[a_off]),
                GoldilocksAvx512::scalar_sub(src[b_off + 1], src[a_off + 1]),
            ];
            let prod = ext2_scalar_mul(diff, challenge, w);

            src[out_off] = GoldilocksAvx512::scalar_add(src[a_off], prod[0]);
            src[out_off + 1] = GoldilocksAvx512::scalar_add(src[a_off + 1], prod[1]);
            i += 1;
        }
    }

    n_pairs * ext_deg
}

// ── Degree-3 extension field reduce ────────────────────────────────────────

/// Degree-3 extension reduce, producing a new Vec (parallel-friendly).
///
/// Each pair of adjacent extension elements `(a, b)` is folded:
/// `result = a + challenge * (b - a)` using degree-3 Karatsuba.
///
/// `src` is `n_elems * 3` u64s in AoS layout. Returns `n_elems/2 * 3` u64s.
#[cfg(feature = "parallel")]
pub fn ext3_reduce_parallel(src: &[u64], challenge: [u64; 3], w: u64) -> Vec<u64> {
    use rayon::prelude::*;

    let ext_deg = 3;
    let pair_u64s = 2 * ext_deg; // 6 u64s per pair (even + odd element)
    let n_pairs = src.len() / pair_u64s;
    let chunk_pairs = 16_384_usize;
    let chunk_u64s = chunk_pairs * pair_u64s;

    if n_pairs <= chunk_pairs {
        return ext3_reduce_chunk(src, challenge, w);
    }

    src.par_chunks(chunk_u64s)
        .flat_map(|chunk| ext3_reduce_chunk(chunk, challenge, w))
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub fn ext3_reduce_parallel(src: &[u64], challenge: [u64; 3], w: u64) -> Vec<u64> {
    ext3_reduce_chunk(src, challenge, w)
}

/// Process a chunk of pairs for ext3 reduce.
fn ext3_reduce_chunk(src: &[u64], challenge: [u64; 3], w: u64) -> Vec<u64> {
    let ext_deg = 3;
    let n_elems = src.len() / ext_deg;
    let n_pairs = n_elems / 2;
    let mut out = vec![0u64; n_pairs * ext_deg];

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_fields::goldilocks::neon::{ext3_scalar_mul, GoldilocksNeon};

        for i in 0..n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            let diff = [
                GoldilocksNeon::scalar_sub(src[b_off], src[a_off]),
                GoldilocksNeon::scalar_sub(src[b_off + 1], src[a_off + 1]),
                GoldilocksNeon::scalar_sub(src[b_off + 2], src[a_off + 2]),
            ];
            let prod = ext3_scalar_mul(diff, challenge, w);
            out[out_off] = GoldilocksNeon::scalar_add(src[a_off], prod[0]);
            out[out_off + 1] = GoldilocksNeon::scalar_add(src[a_off + 1], prod[1]);
            out[out_off + 2] = GoldilocksNeon::scalar_add(src[a_off + 2], prod[2]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        use crate::simd_fields::goldilocks::avx512::{
            ext3_reduce_8pairs, ext3_scalar_mul, GoldilocksAvx512,
        };

        let challenge_v = [
            GoldilocksAvx512::splat(challenge[0]),
            GoldilocksAvx512::splat(challenge[1]),
            GoldilocksAvx512::splat(challenge[2]),
        ];
        let w_vec = GoldilocksAvx512::splat(w);

        // Process 8 pairs at a time (48 input u64s → 24 output u64s)
        let simd_pairs = (n_pairs / 8) * 8;
        let mut i = 0;
        while i < simd_pairs {
            let src_off = (2 * i) * ext_deg;
            let out_off = i * ext_deg;
            unsafe {
                ext3_reduce_8pairs(
                    src.as_ptr().add(src_off),
                    out.as_mut_ptr().add(out_off),
                    challenge_v,
                    w_vec,
                );
            }
            i += 8;
        }

        // Scalar tail for remaining pairs
        while i < n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            let diff = [
                GoldilocksAvx512::scalar_sub(src[b_off], src[a_off]),
                GoldilocksAvx512::scalar_sub(src[b_off + 1], src[a_off + 1]),
                GoldilocksAvx512::scalar_sub(src[b_off + 2], src[a_off + 2]),
            ];
            let prod = ext3_scalar_mul(diff, challenge, w);
            out[out_off] = GoldilocksAvx512::scalar_add(src[a_off], prod[0]);
            out[out_off + 1] = GoldilocksAvx512::scalar_add(src[a_off + 1], prod[1]);
            out[out_off + 2] = GoldilocksAvx512::scalar_add(src[a_off + 2], prod[2]);
            i += 1;
        }
    }

    out
}

/// Degree-3 extension reduce in-place (single-threaded, for small inputs).
#[allow(dead_code)]
pub fn ext3_reduce_in_place<F: SimdBaseField<Scalar = u64>>(
    src: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> usize {
    let ext_deg = 3;
    let n_elems = src.len() / ext_deg;
    let n_pairs = n_elems / 2;

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_fields::goldilocks::neon::{ext3_scalar_mul, GoldilocksNeon};

        for i in 0..n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            let diff = [
                GoldilocksNeon::scalar_sub(src[b_off], src[a_off]),
                GoldilocksNeon::scalar_sub(src[b_off + 1], src[a_off + 1]),
                GoldilocksNeon::scalar_sub(src[b_off + 2], src[a_off + 2]),
            ];
            let prod = ext3_scalar_mul(diff, challenge, w);
            src[out_off] = GoldilocksNeon::scalar_add(src[a_off], prod[0]);
            src[out_off + 1] = GoldilocksNeon::scalar_add(src[a_off + 1], prod[1]);
            src[out_off + 2] = GoldilocksNeon::scalar_add(src[a_off + 2], prod[2]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        use crate::simd_fields::goldilocks::avx512::{
            ext3_reduce_8pairs, ext3_scalar_mul, GoldilocksAvx512,
        };

        let challenge_v = [
            GoldilocksAvx512::splat(challenge[0]),
            GoldilocksAvx512::splat(challenge[1]),
            GoldilocksAvx512::splat(challenge[2]),
        ];
        let w_vec = GoldilocksAvx512::splat(w);

        let ptr = src.as_mut_ptr();
        let simd_pairs = (n_pairs / 8) * 8;
        let mut i = 0;

        // Safe in-place: ext3_reduce_8pairs gathers all 48 u64s into registers
        // before scattering 24 u64s, and output region is always <= input region.
        while i < simd_pairs {
            let src_off = (2 * i) * ext_deg;
            let out_off = i * ext_deg;
            unsafe {
                ext3_reduce_8pairs(
                    ptr.add(src_off) as *const u64,
                    ptr.add(out_off),
                    challenge_v,
                    w_vec,
                );
            }
            i += 8;
        }

        while i < n_pairs {
            let a_off = (2 * i) * ext_deg;
            let b_off = (2 * i + 1) * ext_deg;
            let out_off = i * ext_deg;

            let diff = [
                GoldilocksAvx512::scalar_sub(src[b_off], src[a_off]),
                GoldilocksAvx512::scalar_sub(src[b_off + 1], src[a_off + 1]),
                GoldilocksAvx512::scalar_sub(src[b_off + 2], src[a_off + 2]),
            ];
            let prod = ext3_scalar_mul(diff, challenge, w);
            src[out_off] = GoldilocksAvx512::scalar_add(src[a_off], prod[0]);
            src[out_off + 1] = GoldilocksAvx512::scalar_add(src[a_off + 1], prod[1]);
            src[out_off + 2] = GoldilocksAvx512::scalar_add(src[a_off + 2], prod[2]);
            i += 1;
        }
    }

    n_pairs * ext_deg
}

// ── SoA (Struct-of-Arrays) extension field reduce ─────────────────────────
//
// SoA layout stores each component of an extension field element in a separate
// contiguous array: for ext2 with n elements, c0[0..n] and c1[0..n].
// This eliminates all shuffle overhead (permutex2var, gather/scatter) since
// each component array can be processed with aligned contiguous loads/stores.

/// SoA ext2 reduce in-place.
///
/// Each component array `c0`, `c1` has `len` elements. Adjacent pairs
/// `(elem 2i, elem 2i+1)` are folded: `result = even + challenge * (odd - even)`.
/// The ext2 multiply uses a precomputed `c1*w` for 4 base muls + 2 adds
/// (vs Karatsuba 3 muls + 1 w-mul + 5 adds — same mul count, fewer adds).
///
/// Returns the new length (= len/2).
pub fn ext2_soa_reduce_in_place<F: SimdBaseField<Scalar = u64>>(
    c0: &mut [u64],
    c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> usize {
    let len = c0.len();
    debug_assert_eq!(len, c1.len());
    let n = len / 2;

    let ch0 = F::splat(challenge[0]);
    let ch1 = F::splat(challenge[1]);
    let ch1w = F::splat(F::scalar_mul(challenge[1], w));

    let lanes = F::LANES;
    let step = 4 * lanes;
    let aligned = (n / step) * step;

    let c0_ptr = c0.as_ptr();
    let c1_ptr = c1.as_ptr();
    let c0_out = c0.as_mut_ptr();
    let c1_out = c1.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            for g in 0..4 {
                let off = i + g * lanes;
                let (c0_even, c0_odd) = F::load_deinterleaved(c0_ptr.add(2 * off));
                let (c1_even, c1_odd) = F::load_deinterleaved(c1_ptr.add(2 * off));

                let d0 = F::sub(c0_odd, c0_even);
                let d1 = F::sub(c1_odd, c1_even);

                // challenge * diff = (ch0*d0 + ch1w*d1, ch0*d1 + ch1*d0)
                let prod_c0 = F::add(F::mul(ch0, d0), F::mul(ch1w, d1));
                let prod_c1 = F::add(F::mul(ch0, d1), F::mul(ch1, d0));

                F::store(c0_out.add(off), F::add(c0_even, prod_c0));
                F::store(c1_out.add(off), F::add(c1_even, prod_c1));
            }
        }
        i += step;
    }

    while i + lanes <= n {
        unsafe {
            let (c0_even, c0_odd) = F::load_deinterleaved(c0_ptr.add(2 * i));
            let (c1_even, c1_odd) = F::load_deinterleaved(c1_ptr.add(2 * i));

            let d0 = F::sub(c0_odd, c0_even);
            let d1 = F::sub(c1_odd, c1_even);

            let prod_c0 = F::add(F::mul(ch0, d0), F::mul(ch1w, d1));
            let prod_c1 = F::add(F::mul(ch0, d1), F::mul(ch1, d0));

            F::store(c0_out.add(i), F::add(c0_even, prod_c0));
            F::store(c1_out.add(i), F::add(c1_even, prod_c1));
        }
        i += lanes;
    }

    // Scalar tail
    let ch1w_s = F::scalar_mul(challenge[1], w);
    while i < n {
        let d0 = F::scalar_sub(c0[2 * i + 1], c0[2 * i]);
        let d1 = F::scalar_sub(c1[2 * i + 1], c1[2 * i]);

        let prod_c0 = F::scalar_add(F::scalar_mul(challenge[0], d0), F::scalar_mul(ch1w_s, d1));
        let prod_c1 = F::scalar_add(
            F::scalar_mul(challenge[0], d1),
            F::scalar_mul(challenge[1], d0),
        );

        c0[i] = F::scalar_add(c0[2 * i], prod_c0);
        c1[i] = F::scalar_add(c1[2 * i], prod_c1);
        i += 1;
    }

    n
}

/// Fused SoA ext2 reduce + next-round evaluate in a single pass.
///
/// Reduces pairs in-place and simultaneously accumulates even/odd component sums
/// for the next round's evaluate, eliminating one full data pass per round.
/// Uses lazy accumulation (wrapping add + carry) for cheap accumulation.
///
/// Returns `(even_components, odd_components, new_len)`.
pub fn ext2_soa_reduce_and_evaluate<F: SimdBaseField<Scalar = u64>>(
    c0: &mut [u64],
    c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2], usize) {
    let len = c0.len();
    debug_assert_eq!(len, c1.len());
    let n = len / 2;

    // SAFETY: single-threaded ascending iteration is safe in-place because
    // reads at src[2i, 2i+1] precede writes at out[i] for each step i.
    let (even, odd) = unsafe {
        ext2_soa_reduce_and_evaluate_raw::<F>(
            c0.as_ptr(),
            c1.as_ptr(),
            c0.as_mut_ptr(),
            c1.as_mut_ptr(),
            n,
            challenge,
            w,
        )
    };
    (even, odd, n)
}

/// Distinct-buffer version of `ext2_soa_reduce_and_evaluate`.
///
/// Reads from `src_c0`/`src_c1` (length `2 * n`) and writes to
/// `out_c0`/`out_c1` (length `n`). Used by the parallel chunked kernel.
pub fn ext2_soa_reduce_and_evaluate_into<F: SimdBaseField<Scalar = u64>>(
    src_c0: &[u64],
    src_c1: &[u64],
    out_c0: &mut [u64],
    out_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    let n = out_c0.len();
    debug_assert_eq!(n, out_c1.len());
    debug_assert_eq!(src_c0.len(), 2 * n);
    debug_assert_eq!(src_c1.len(), 2 * n);
    unsafe {
        ext2_soa_reduce_and_evaluate_raw::<F>(
            src_c0.as_ptr(),
            src_c1.as_ptr(),
            out_c0.as_mut_ptr(),
            out_c1.as_mut_ptr(),
            n,
            challenge,
            w,
        )
    }
}

/// Raw-pointer core of `ext2_soa_reduce_and_evaluate`.
///
/// # Safety
/// - `src_c0_ptr` / `src_c1_ptr` must each be valid for reading `2 * n` u64s.
/// - `out_c0_ptr` / `out_c1_ptr` must each be valid for writing `n` u64s.
/// - If src and out alias the same buffer, the caller must use single-threaded
///   ascending iteration (read `[2i, 2i+1]` happens before write `[i]` per i).
///   Parallel chunked callers must pass non-overlapping src/out regions.
#[inline(always)]
unsafe fn ext2_soa_reduce_and_evaluate_raw<F: SimdBaseField<Scalar = u64>>(
    src_c0_ptr: *const u64,
    src_c1_ptr: *const u64,
    out_c0_ptr: *mut u64,
    out_c1_ptr: *mut u64,
    n: usize,
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    let ch0 = F::splat(challenge[0]);
    let ch1 = F::splat(challenge[1]);
    let ch1w = F::splat(F::scalar_mul(challenge[1], w));

    let lanes = F::LANES;
    let step = 2 * lanes; // 2× unroll
    let aligned = (n / step) * step;

    // Lazy accumulators: 2 per component × 2 unroll groups
    let zero = F::splat(F::ZERO);
    let mut acc_c0_0 = zero;
    let mut acc_c0_1 = zero;
    let mut acc_c1_0 = zero;
    let mut acc_c1_1 = zero;
    let mut carry_c0_0 = zero;
    let mut carry_c0_1 = zero;
    let mut carry_c1_0 = zero;
    let mut carry_c1_1 = zero;

    let mut i = 0;
    while i < aligned {
        // Group 0
        let off0 = i;
        let (e0_0, o0_0) = F::load_deinterleaved(src_c0_ptr.add(2 * off0));
        let (e1_0, o1_0) = F::load_deinterleaved(src_c1_ptr.add(2 * off0));

        let d0_0 = F::sub(o0_0, e0_0);
        let d1_0 = F::sub(o1_0, e1_0);
        let r0_0 = F::add(e0_0, F::add(F::mul(ch0, d0_0), F::mul(ch1w, d1_0)));
        let r1_0 = F::add(e1_0, F::add(F::mul(ch0, d1_0), F::mul(ch1, d0_0)));

        F::store(out_c0_ptr.add(off0), r0_0);
        F::store(out_c1_ptr.add(off0), r1_0);

        let s = F::add_wrapping(acc_c0_0, r0_0);
        carry_c0_0 = F::add_wrapping(carry_c0_0, F::carry_mask(s, acc_c0_0));
        acc_c0_0 = s;
        let s = F::add_wrapping(acc_c1_0, r1_0);
        carry_c1_0 = F::add_wrapping(carry_c1_0, F::carry_mask(s, acc_c1_0));
        acc_c1_0 = s;

        // Group 1
        let off1 = i + lanes;
        let (e0_1, o0_1) = F::load_deinterleaved(src_c0_ptr.add(2 * off1));
        let (e1_1, o1_1) = F::load_deinterleaved(src_c1_ptr.add(2 * off1));

        let d0_1 = F::sub(o0_1, e0_1);
        let d1_1 = F::sub(o1_1, e1_1);
        let r0_1 = F::add(e0_1, F::add(F::mul(ch0, d0_1), F::mul(ch1w, d1_1)));
        let r1_1 = F::add(e1_1, F::add(F::mul(ch0, d1_1), F::mul(ch1, d0_1)));

        F::store(out_c0_ptr.add(off1), r0_1);
        F::store(out_c1_ptr.add(off1), r1_1);

        let s = F::add_wrapping(acc_c0_1, r0_1);
        carry_c0_1 = F::add_wrapping(carry_c0_1, F::carry_mask(s, acc_c0_1));
        acc_c0_1 = s;
        let s = F::add_wrapping(acc_c1_1, r1_1);
        carry_c1_1 = F::add_wrapping(carry_c1_1, F::carry_mask(s, acc_c1_1));
        acc_c1_1 = s;
        i += step;
    }

    // Cleanup: single vector at a time with full modular add
    while i + lanes <= n {
        let (e0, o0) = F::load_deinterleaved(src_c0_ptr.add(2 * i));
        let (e1, o1) = F::load_deinterleaved(src_c1_ptr.add(2 * i));

        let d0 = F::sub(o0, e0);
        let d1 = F::sub(o1, e1);
        let r0 = F::add(e0, F::add(F::mul(ch0, d0), F::mul(ch1w, d1)));
        let r1 = F::add(e1, F::add(F::mul(ch0, d1), F::mul(ch1, d0)));

        F::store(out_c0_ptr.add(i), r0);
        F::store(out_c1_ptr.add(i), r1);
        acc_c0_0 = F::add(acc_c0_0, r0);
        acc_c1_0 = F::add(acc_c1_0, r1);
        i += lanes;
    }

    // Finalize lazy accumulators
    let total_c0 = F::add(
        F::reduce_carry(acc_c0_0, carry_c0_0),
        F::reduce_carry(acc_c0_1, carry_c0_1),
    );
    let total_c1 = F::add(
        F::reduce_carry(acc_c1_0, carry_c1_0),
        F::reduce_carry(acc_c1_1, carry_c1_1),
    );

    // Extract even/odd lanes
    let mut buf = [F::ZERO; 32];
    let mut even = [F::ZERO; 2];
    let mut odd = [F::ZERO; 2];

    F::store(buf.as_mut_ptr(), total_c0);
    for (j, &v) in buf.iter().enumerate().take(F::LANES) {
        if j % 2 == 0 {
            even[0] = F::scalar_add(even[0], v);
        } else {
            odd[0] = F::scalar_add(odd[0], v);
        }
    }
    F::store(buf.as_mut_ptr(), total_c1);
    for (j, &v) in buf.iter().enumerate().take(F::LANES) {
        if j % 2 == 0 {
            even[1] = F::scalar_add(even[1], v);
        } else {
            odd[1] = F::scalar_add(odd[1], v);
        }
    }

    // Scalar tail
    let ch1w_s = F::scalar_mul(challenge[1], w);
    while i < n {
        let a0 = *src_c0_ptr.add(2 * i);
        let b0 = *src_c0_ptr.add(2 * i + 1);
        let a1 = *src_c1_ptr.add(2 * i);
        let b1 = *src_c1_ptr.add(2 * i + 1);

        let d0 = F::scalar_sub(b0, a0);
        let d1 = F::scalar_sub(b1, a1);

        let r0 = F::scalar_add(
            a0,
            F::scalar_add(F::scalar_mul(challenge[0], d0), F::scalar_mul(ch1w_s, d1)),
        );
        let r1 = F::scalar_add(
            a1,
            F::scalar_add(
                F::scalar_mul(challenge[0], d1),
                F::scalar_mul(challenge[1], d0),
            ),
        );

        *out_c0_ptr.add(i) = r0;
        *out_c1_ptr.add(i) = r1;

        if i % 2 == 0 {
            even[0] = F::scalar_add(even[0], r0);
            even[1] = F::scalar_add(even[1], r1);
        } else {
            odd[0] = F::scalar_add(odd[0], r0);
            odd[1] = F::scalar_add(odd[1], r1);
        }
        i += 1;
    }

    (even, odd)
}

/// Parallel fused SoA ext2 reduce + next-round evaluate.
///
/// Splits the output into rayon chunks and processes each chunk with
/// `ext2_soa_reduce_and_evaluate_raw` on distinct src/out regions.
///
/// `chunk_pairs` must be even so each chunk starts at an even global pair
/// index (preserving even/odd lane parity in horizontal reductions).
#[cfg(feature = "parallel")]
pub fn ext2_soa_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_c0: &[u64],
    src_c1: &[u64],
    out_c0: &mut [u64],
    out_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    use rayon::prelude::*;

    let n = out_c0.len();
    debug_assert_eq!(n, out_c1.len());
    debug_assert_eq!(src_c0.len(), 2 * n);
    debug_assert_eq!(src_c1.len(), 2 * n);

    let chunk_pairs = 32_768_usize; // power of 2, multiple of 2*LANES, even
    if n <= chunk_pairs {
        return ext2_soa_reduce_and_evaluate_into::<F>(
            src_c0, src_c1, out_c0, out_c1, challenge, w,
        );
    }

    out_c0
        .par_chunks_mut(chunk_pairs)
        .zip(out_c1.par_chunks_mut(chunk_pairs))
        .enumerate()
        .map(|(idx, (oc0, oc1))| {
            let start = idx * chunk_pairs;
            let end = start + oc0.len();
            ext2_soa_reduce_and_evaluate_into::<F>(
                &src_c0[2 * start..2 * end],
                &src_c1[2 * start..2 * end],
                oc0,
                oc1,
                challenge,
                w,
            )
        })
        .reduce(
            || ([0u64; 2], [0u64; 2]),
            |(e1, o1), (e2, o2)| {
                (
                    [F::scalar_add(e1[0], e2[0]), F::scalar_add(e1[1], e2[1])],
                    [F::scalar_add(o1[0], o2[0]), F::scalar_add(o1[1], o2[1])],
                )
            },
        )
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
pub fn ext2_soa_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_c0: &[u64],
    src_c1: &[u64],
    out_c0: &mut [u64],
    out_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    ext2_soa_reduce_and_evaluate_into::<F>(src_c0, src_c1, out_c0, out_c1, challenge, w)
}

/// SoA ext3 reduce in-place.
///
/// Same concept as ext2 but for degree-3 extensions.
/// Uses Karatsuba multiplication: 6 base muls + 2 mul-by-w + adds.
/// Returns the new length (= len/2).
pub fn ext3_soa_reduce_in_place<F: SimdBaseField<Scalar = u64>>(
    c0: &mut [u64],
    c1: &mut [u64],
    c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> usize {
    let len = c0.len();
    debug_assert_eq!(len, c1.len());
    debug_assert_eq!(len, c2.len());
    let n = len / 2;

    let ch = [
        F::splat(challenge[0]),
        F::splat(challenge[1]),
        F::splat(challenge[2]),
    ];
    let w_vec = F::splat(w);

    let lanes = F::LANES;
    let step = 2 * lanes; // 2× unroll (more register pressure with ext3)
    let aligned = (n / step) * step;

    let c0_ptr = c0.as_ptr();
    let c1_ptr = c1.as_ptr();
    let c2_ptr = c2.as_ptr();
    let c0_out = c0.as_mut_ptr();
    let c1_out = c1.as_mut_ptr();
    let c2_out = c2.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            for g in 0..2 {
                let off = i + g * lanes;
                let (e0, o0) = F::load_deinterleaved(c0_ptr.add(2 * off));
                let (e1, o1) = F::load_deinterleaved(c1_ptr.add(2 * off));
                let (e2, o2) = F::load_deinterleaved(c2_ptr.add(2 * off));

                let d = [F::sub(o0, e0), F::sub(o1, e1), F::sub(o2, e2)];

                // Karatsuba ext3: challenge * diff
                let ad = F::mul(ch[0], d[0]);
                let be = F::mul(ch[1], d[1]);
                let cf = F::mul(ch[2], d[2]);

                let x = F::sub(
                    F::sub(F::mul(F::add(ch[1], ch[2]), F::add(d[1], d[2])), be),
                    cf,
                );
                let y = F::sub(
                    F::sub(F::mul(F::add(ch[0], ch[1]), F::add(d[0], d[1])), ad),
                    be,
                );
                let z = F::add(
                    F::sub(
                        F::sub(F::mul(F::add(ch[0], ch[2]), F::add(d[0], d[2])), ad),
                        cf,
                    ),
                    be,
                );

                let r0 = F::add(ad, F::mul(w_vec, x));
                let r1 = F::add(y, F::mul(w_vec, cf));
                let r2 = z;

                F::store(c0_out.add(off), F::add(e0, r0));
                F::store(c1_out.add(off), F::add(e1, r1));
                F::store(c2_out.add(off), F::add(e2, r2));
            }
        }
        i += step;
    }

    while i + lanes <= n {
        unsafe {
            let (e0, o0) = F::load_deinterleaved(c0_ptr.add(2 * i));
            let (e1, o1) = F::load_deinterleaved(c1_ptr.add(2 * i));
            let (e2, o2) = F::load_deinterleaved(c2_ptr.add(2 * i));

            let d = [F::sub(o0, e0), F::sub(o1, e1), F::sub(o2, e2)];

            let ad = F::mul(ch[0], d[0]);
            let be = F::mul(ch[1], d[1]);
            let cf = F::mul(ch[2], d[2]);

            let x = F::sub(
                F::sub(F::mul(F::add(ch[1], ch[2]), F::add(d[1], d[2])), be),
                cf,
            );
            let y = F::sub(
                F::sub(F::mul(F::add(ch[0], ch[1]), F::add(d[0], d[1])), ad),
                be,
            );
            let z = F::add(
                F::sub(
                    F::sub(F::mul(F::add(ch[0], ch[2]), F::add(d[0], d[2])), ad),
                    cf,
                ),
                be,
            );

            F::store(c0_out.add(i), F::add(e0, F::add(ad, F::mul(w_vec, x))));
            F::store(c1_out.add(i), F::add(e1, F::add(y, F::mul(w_vec, cf))));
            F::store(c2_out.add(i), F::add(e2, z));
        }
        i += lanes;
    }

    // Scalar tail
    while i < n {
        let d = [
            F::scalar_sub(c0[2 * i + 1], c0[2 * i]),
            F::scalar_sub(c1[2 * i + 1], c1[2 * i]),
            F::scalar_sub(c2[2 * i + 1], c2[2 * i]),
        ];

        let ad = F::scalar_mul(challenge[0], d[0]);
        let be = F::scalar_mul(challenge[1], d[1]);
        let cf = F::scalar_mul(challenge[2], d[2]);

        let x = F::scalar_sub(
            F::scalar_sub(
                F::scalar_mul(
                    F::scalar_add(challenge[1], challenge[2]),
                    F::scalar_add(d[1], d[2]),
                ),
                be,
            ),
            cf,
        );
        let y = F::scalar_sub(
            F::scalar_sub(
                F::scalar_mul(
                    F::scalar_add(challenge[0], challenge[1]),
                    F::scalar_add(d[0], d[1]),
                ),
                ad,
            ),
            be,
        );
        let z = F::scalar_add(
            F::scalar_sub(
                F::scalar_sub(
                    F::scalar_mul(
                        F::scalar_add(challenge[0], challenge[2]),
                        F::scalar_add(d[0], d[2]),
                    ),
                    ad,
                ),
                cf,
            ),
            be,
        );

        c0[i] = F::scalar_add(c0[2 * i], F::scalar_add(ad, F::scalar_mul(w, x)));
        c1[i] = F::scalar_add(c1[2 * i], F::scalar_add(y, F::scalar_mul(w, cf)));
        c2[i] = F::scalar_add(c2[2 * i], z);
        i += 1;
    }

    n
}

/// Fused SoA ext3 reduce + next-round evaluate in a single pass.
///
/// Same concept as ext2 fused kernel but with Karatsuba ext3 multiply.
/// 1x unroll due to higher register pressure (3 components × 2 accum × 2 carry = 12 zmm).
///
/// Returns `(even_components, odd_components, new_len)`.
pub fn ext3_soa_reduce_and_evaluate<F: SimdBaseField<Scalar = u64>>(
    c0: &mut [u64],
    c1: &mut [u64],
    c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3], usize) {
    let len = c0.len();
    debug_assert_eq!(len, c1.len());
    debug_assert_eq!(len, c2.len());
    let n = len / 2;

    // SAFETY: single-threaded ascending iteration is safe in-place.
    let (even, odd) = unsafe {
        ext3_soa_reduce_and_evaluate_raw::<F>(
            c0.as_ptr(),
            c1.as_ptr(),
            c2.as_ptr(),
            c0.as_mut_ptr(),
            c1.as_mut_ptr(),
            c2.as_mut_ptr(),
            n,
            challenge,
            w,
        )
    };
    (even, odd, n)
}

/// Distinct-buffer version of `ext3_soa_reduce_and_evaluate`.
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_reduce_and_evaluate_into<F: SimdBaseField<Scalar = u64>>(
    src_c0: &[u64],
    src_c1: &[u64],
    src_c2: &[u64],
    out_c0: &mut [u64],
    out_c1: &mut [u64],
    out_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    let n = out_c0.len();
    debug_assert_eq!(n, out_c1.len());
    debug_assert_eq!(n, out_c2.len());
    debug_assert_eq!(src_c0.len(), 2 * n);
    debug_assert_eq!(src_c1.len(), 2 * n);
    debug_assert_eq!(src_c2.len(), 2 * n);
    unsafe {
        ext3_soa_reduce_and_evaluate_raw::<F>(
            src_c0.as_ptr(),
            src_c1.as_ptr(),
            src_c2.as_ptr(),
            out_c0.as_mut_ptr(),
            out_c1.as_mut_ptr(),
            out_c2.as_mut_ptr(),
            n,
            challenge,
            w,
        )
    }
}

/// Raw-pointer core of `ext3_soa_reduce_and_evaluate`.
///
/// # Safety
/// Same contract as `ext2_soa_reduce_and_evaluate_raw`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ext3_soa_reduce_and_evaluate_raw<F: SimdBaseField<Scalar = u64>>(
    src_c0_ptr: *const u64,
    src_c1_ptr: *const u64,
    src_c2_ptr: *const u64,
    out_c0_ptr: *mut u64,
    out_c1_ptr: *mut u64,
    out_c2_ptr: *mut u64,
    n: usize,
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    let ch = [
        F::splat(challenge[0]),
        F::splat(challenge[1]),
        F::splat(challenge[2]),
    ];
    let w_vec = F::splat(w);

    let lanes = F::LANES;
    let aligned = (n / lanes) * lanes;

    let zero = F::splat(F::ZERO);
    let mut acc = [zero; 3];
    let mut carry = [zero; 3];

    let mut i = 0;
    while i < aligned {
        let (e0, o0) = F::load_deinterleaved(src_c0_ptr.add(2 * i));
        let (e1, o1) = F::load_deinterleaved(src_c1_ptr.add(2 * i));
        let (e2, o2) = F::load_deinterleaved(src_c2_ptr.add(2 * i));

        let d = [F::sub(o0, e0), F::sub(o1, e1), F::sub(o2, e2)];

        // Karatsuba ext3: challenge * diff
        let ad = F::mul(ch[0], d[0]);
        let be = F::mul(ch[1], d[1]);
        let cf = F::mul(ch[2], d[2]);

        let x = F::sub(
            F::sub(F::mul(F::add(ch[1], ch[2]), F::add(d[1], d[2])), be),
            cf,
        );
        let y = F::sub(
            F::sub(F::mul(F::add(ch[0], ch[1]), F::add(d[0], d[1])), ad),
            be,
        );
        let z = F::add(
            F::sub(
                F::sub(F::mul(F::add(ch[0], ch[2]), F::add(d[0], d[2])), ad),
                cf,
            ),
            be,
        );

        let r0 = F::add(e0, F::add(ad, F::mul(w_vec, x)));
        let r1 = F::add(e1, F::add(y, F::mul(w_vec, cf)));
        let r2 = F::add(e2, z);

        F::store(out_c0_ptr.add(i), r0);
        F::store(out_c1_ptr.add(i), r1);
        F::store(out_c2_ptr.add(i), r2);

        let s0 = F::add_wrapping(acc[0], r0);
        carry[0] = F::add_wrapping(carry[0], F::carry_mask(s0, acc[0]));
        acc[0] = s0;
        let s1 = F::add_wrapping(acc[1], r1);
        carry[1] = F::add_wrapping(carry[1], F::carry_mask(s1, acc[1]));
        acc[1] = s1;
        let s2 = F::add_wrapping(acc[2], r2);
        carry[2] = F::add_wrapping(carry[2], F::carry_mask(s2, acc[2]));
        acc[2] = s2;
        i += lanes;
    }

    // Finalize
    let total = [
        F::reduce_carry(acc[0], carry[0]),
        F::reduce_carry(acc[1], carry[1]),
        F::reduce_carry(acc[2], carry[2]),
    ];

    let mut buf = [F::ZERO; 32];
    let mut even = [F::ZERO; 3];
    let mut odd = [F::ZERO; 3];

    for c in 0..3 {
        F::store(buf.as_mut_ptr(), total[c]);
        for (j, &v) in buf.iter().enumerate().take(F::LANES) {
            if j % 2 == 0 {
                even[c] = F::scalar_add(even[c], v);
            } else {
                odd[c] = F::scalar_add(odd[c], v);
            }
        }
    }

    // Scalar tail
    while i < n {
        let a0 = *src_c0_ptr.add(2 * i);
        let b0 = *src_c0_ptr.add(2 * i + 1);
        let a1 = *src_c1_ptr.add(2 * i);
        let b1 = *src_c1_ptr.add(2 * i + 1);
        let a2 = *src_c2_ptr.add(2 * i);
        let b2 = *src_c2_ptr.add(2 * i + 1);

        let d = [
            F::scalar_sub(b0, a0),
            F::scalar_sub(b1, a1),
            F::scalar_sub(b2, a2),
        ];

        let ad = F::scalar_mul(challenge[0], d[0]);
        let be = F::scalar_mul(challenge[1], d[1]);
        let cf = F::scalar_mul(challenge[2], d[2]);
        let x = F::scalar_sub(
            F::scalar_sub(
                F::scalar_mul(
                    F::scalar_add(challenge[1], challenge[2]),
                    F::scalar_add(d[1], d[2]),
                ),
                be,
            ),
            cf,
        );
        let y = F::scalar_sub(
            F::scalar_sub(
                F::scalar_mul(
                    F::scalar_add(challenge[0], challenge[1]),
                    F::scalar_add(d[0], d[1]),
                ),
                ad,
            ),
            be,
        );
        let z = F::scalar_add(
            F::scalar_sub(
                F::scalar_sub(
                    F::scalar_mul(
                        F::scalar_add(challenge[0], challenge[2]),
                        F::scalar_add(d[0], d[2]),
                    ),
                    ad,
                ),
                cf,
            ),
            be,
        );

        let r = [
            F::scalar_add(a0, F::scalar_add(ad, F::scalar_mul(w, x))),
            F::scalar_add(a1, F::scalar_add(y, F::scalar_mul(w, cf))),
            F::scalar_add(a2, z),
        ];
        *out_c0_ptr.add(i) = r[0];
        *out_c1_ptr.add(i) = r[1];
        *out_c2_ptr.add(i) = r[2];

        if i % 2 == 0 {
            for c in 0..3 {
                even[c] = F::scalar_add(even[c], r[c]);
            }
        } else {
            for c in 0..3 {
                odd[c] = F::scalar_add(odd[c], r[c]);
            }
        }
        i += 1;
    }

    (even, odd)
}

/// Parallel fused SoA ext3 reduce + next-round evaluate.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_c0: &[u64],
    src_c1: &[u64],
    src_c2: &[u64],
    out_c0: &mut [u64],
    out_c1: &mut [u64],
    out_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    use rayon::prelude::*;

    let n = out_c0.len();
    let chunk_pairs = 32_768_usize;
    if n <= chunk_pairs {
        return ext3_soa_reduce_and_evaluate_into::<F>(
            src_c0, src_c1, src_c2, out_c0, out_c1, out_c2, challenge, w,
        );
    }

    // Split all three output components in parallel. Since rayon's par_chunks_mut
    // only takes a single slice, we zip three separate par_chunks_mut iterators.
    (out_c0.par_chunks_mut(chunk_pairs))
        .zip(out_c1.par_chunks_mut(chunk_pairs))
        .zip(out_c2.par_chunks_mut(chunk_pairs))
        .enumerate()
        .map(|(idx, ((oc0, oc1), oc2))| {
            let start = idx * chunk_pairs;
            let end = start + oc0.len();
            ext3_soa_reduce_and_evaluate_into::<F>(
                &src_c0[2 * start..2 * end],
                &src_c1[2 * start..2 * end],
                &src_c2[2 * start..2 * end],
                oc0,
                oc1,
                oc2,
                challenge,
                w,
            )
        })
        .reduce(
            || ([0u64; 3], [0u64; 3]),
            |(e1, o1), (e2, o2)| {
                (
                    [
                        F::scalar_add(e1[0], e2[0]),
                        F::scalar_add(e1[1], e2[1]),
                        F::scalar_add(e1[2], e2[2]),
                    ],
                    [
                        F::scalar_add(o1[0], o2[0]),
                        F::scalar_add(o1[1], o2[1]),
                        F::scalar_add(o1[2], o2[2]),
                    ],
                )
            },
        )
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_c0: &[u64],
    src_c1: &[u64],
    src_c2: &[u64],
    out_c0: &mut [u64],
    out_c1: &mut [u64],
    out_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    ext3_soa_reduce_and_evaluate_into::<F>(
        src_c0, src_c1, src_c2, out_c0, out_c1, out_c2, challenge, w,
    )
}

/// Fused SoA ext2 product evaluate + reduce in a single pass.
///
/// Computes the inner product evaluate (a, b) AND reduces both f and g in one
/// streaming pass over the data. Eliminates 2 full data passes per round.
///
/// Returns `(a_components, b_components, new_len)`.
pub fn ext2_soa_product_reduce_and_evaluate<F: SimdBaseField<Scalar = u64>>(
    f_c0: &mut [u64],
    f_c1: &mut [u64],
    g_c0: &mut [u64],
    g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2], usize) {
    let n = f_c0.len();
    debug_assert_eq!(n, f_c1.len());
    debug_assert_eq!(n, g_c0.len());
    debug_assert_eq!(n, g_c1.len());
    let half = n / 2;

    // SAFETY: single-threaded ascending iteration is safe in-place.
    let (a, b) = unsafe {
        ext2_soa_product_reduce_and_evaluate_raw::<F>(
            f_c0.as_ptr(),
            f_c1.as_ptr(),
            g_c0.as_ptr(),
            g_c1.as_ptr(),
            f_c0.as_mut_ptr(),
            f_c1.as_mut_ptr(),
            g_c0.as_mut_ptr(),
            g_c1.as_mut_ptr(),
            half,
            challenge,
            w,
        )
    };
    (a, b, half)
}

/// Distinct-buffer version of `ext2_soa_product_reduce_and_evaluate`.
#[allow(clippy::too_many_arguments)]
pub fn ext2_soa_product_reduce_and_evaluate_into<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    let n_out = out_f_c0.len();
    debug_assert_eq!(n_out, out_f_c1.len());
    debug_assert_eq!(n_out, out_g_c0.len());
    debug_assert_eq!(n_out, out_g_c1.len());
    debug_assert_eq!(src_f_c0.len(), 2 * n_out);
    debug_assert_eq!(src_f_c1.len(), 2 * n_out);
    debug_assert_eq!(src_g_c0.len(), 2 * n_out);
    debug_assert_eq!(src_g_c1.len(), 2 * n_out);
    unsafe {
        ext2_soa_product_reduce_and_evaluate_raw::<F>(
            src_f_c0.as_ptr(),
            src_f_c1.as_ptr(),
            src_g_c0.as_ptr(),
            src_g_c1.as_ptr(),
            out_f_c0.as_mut_ptr(),
            out_f_c1.as_mut_ptr(),
            out_g_c0.as_mut_ptr(),
            out_g_c1.as_mut_ptr(),
            n_out,
            challenge,
            w,
        )
    }
}

/// Raw-pointer core of `ext2_soa_product_reduce_and_evaluate`.
///
/// # Safety
/// Same contract as `ext2_soa_reduce_and_evaluate_raw`, but with both f and g.
/// `n_out` is the number of output pairs (input has `2 * n_out` elements per slice).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ext2_soa_product_reduce_and_evaluate_raw<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: *const u64,
    src_f_c1: *const u64,
    src_g_c0: *const u64,
    src_g_c1: *const u64,
    out_f_c0: *mut u64,
    out_f_c1: *mut u64,
    out_g_c0: *mut u64,
    out_g_c1: *mut u64,
    n_out: usize,
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    let lanes = F::LANES;
    let aligned = (n_out / lanes) * lanes;
    let w_vec = F::splat(w);
    let ch0 = F::splat(challenge[0]);
    let ch1 = F::splat(challenge[1]);
    let ch1w = F::splat(F::scalar_mul(challenge[1], w));

    let zero = F::splat(F::ZERO);
    let mut acc_a0 = zero;
    let mut acc_a1 = zero;
    let mut acc_b0 = zero;
    let mut acc_b1 = zero;

    let mut i = 0;
    while i < aligned {
        let off = i;
        let (fe0, fo0) = F::load_deinterleaved(src_f_c0.add(2 * off));
        let (fe1, fo1) = F::load_deinterleaved(src_f_c1.add(2 * off));
        let (ge0, go0) = F::load_deinterleaved(src_g_c0.add(2 * off));
        let (ge1, go1) = F::load_deinterleaved(src_g_c1.add(2 * off));

        // a += f_even * g_even (ext2 Karatsuba)
        let v0 = F::mul(fe0, ge0);
        let v1 = F::mul(fe1, ge1);
        acc_a0 = F::add(acc_a0, F::add(v0, F::mul(w_vec, v1)));
        let m = F::mul(F::add(fe0, fe1), F::add(ge0, ge1));
        acc_a1 = F::add(acc_a1, F::sub(F::sub(m, v0), v1));

        // b += f_even * g_odd + f_odd * g_even
        let u0 = F::mul(fe0, go0);
        let u1 = F::mul(fe1, go1);
        let m1 = F::mul(F::add(fe0, fe1), F::add(go0, go1));
        let p0 = F::mul(fo0, ge0);
        let p1 = F::mul(fo1, ge1);
        let m2 = F::mul(F::add(fo0, fo1), F::add(ge0, ge1));

        acc_b0 = F::add(
            acc_b0,
            F::add(F::add(u0, F::mul(w_vec, u1)), F::add(p0, F::mul(w_vec, p1))),
        );
        acc_b1 = F::add(
            acc_b1,
            F::add(F::sub(F::sub(m1, u0), u1), F::sub(F::sub(m2, p0), p1)),
        );

        // Reduce f
        let fd0 = F::sub(fo0, fe0);
        let fd1 = F::sub(fo1, fe1);
        F::store(
            out_f_c0.add(off),
            F::add(fe0, F::add(F::mul(ch0, fd0), F::mul(ch1w, fd1))),
        );
        F::store(
            out_f_c1.add(off),
            F::add(fe1, F::add(F::mul(ch0, fd1), F::mul(ch1, fd0))),
        );

        // Reduce g
        let gd0 = F::sub(go0, ge0);
        let gd1 = F::sub(go1, ge1);
        F::store(
            out_g_c0.add(off),
            F::add(ge0, F::add(F::mul(ch0, gd0), F::mul(ch1w, gd1))),
        );
        F::store(
            out_g_c1.add(off),
            F::add(ge1, F::add(F::mul(ch0, gd1), F::mul(ch1, gd0))),
        );
        i += lanes;
    }

    // Horizontal reduce
    let mut buf = [F::ZERO; 32];
    let mut a = [F::ZERO; 2];
    let mut b = [F::ZERO; 2];

    F::store(buf.as_mut_ptr(), acc_a0);
    for &v in buf.iter().take(lanes) {
        a[0] = F::scalar_add(a[0], v);
    }
    F::store(buf.as_mut_ptr(), acc_a1);
    for &v in buf.iter().take(lanes) {
        a[1] = F::scalar_add(a[1], v);
    }
    F::store(buf.as_mut_ptr(), acc_b0);
    for &v in buf.iter().take(lanes) {
        b[0] = F::scalar_add(b[0], v);
    }
    F::store(buf.as_mut_ptr(), acc_b1);
    for &v in buf.iter().take(lanes) {
        b[1] = F::scalar_add(b[1], v);
    }

    // Scalar tail
    let ch1w_s = F::scalar_mul(challenge[1], w);
    while i < n_out {
        let fe = [*src_f_c0.add(2 * i), *src_f_c1.add(2 * i)];
        let fo = [*src_f_c0.add(2 * i + 1), *src_f_c1.add(2 * i + 1)];
        let ge = [*src_g_c0.add(2 * i), *src_g_c1.add(2 * i)];
        let go_ = [*src_g_c0.add(2 * i + 1), *src_g_c1.add(2 * i + 1)];

        let v0 = F::scalar_mul(fe[0], ge[0]);
        let v1 = F::scalar_mul(fe[1], ge[1]);
        a[0] = F::scalar_add(a[0], F::scalar_add(v0, F::scalar_mul(w, v1)));
        let m = F::scalar_mul(F::scalar_add(fe[0], fe[1]), F::scalar_add(ge[0], ge[1]));
        a[1] = F::scalar_add(a[1], F::scalar_sub(F::scalar_sub(m, v0), v1));

        let u0 = F::scalar_mul(fe[0], go_[0]);
        let u1 = F::scalar_mul(fe[1], go_[1]);
        let m1 = F::scalar_mul(F::scalar_add(fe[0], fe[1]), F::scalar_add(go_[0], go_[1]));
        let p0 = F::scalar_mul(fo[0], ge[0]);
        let p1 = F::scalar_mul(fo[1], ge[1]);
        let m2 = F::scalar_mul(F::scalar_add(fo[0], fo[1]), F::scalar_add(ge[0], ge[1]));
        b[0] = F::scalar_add(
            b[0],
            F::scalar_add(
                F::scalar_add(u0, F::scalar_mul(w, u1)),
                F::scalar_add(p0, F::scalar_mul(w, p1)),
            ),
        );
        b[1] = F::scalar_add(
            b[1],
            F::scalar_add(
                F::scalar_sub(F::scalar_sub(m1, u0), u1),
                F::scalar_sub(F::scalar_sub(m2, p0), p1),
            ),
        );

        let fd0 = F::scalar_sub(fo[0], fe[0]);
        let fd1 = F::scalar_sub(fo[1], fe[1]);
        *out_f_c0.add(i) = F::scalar_add(
            fe[0],
            F::scalar_add(F::scalar_mul(challenge[0], fd0), F::scalar_mul(ch1w_s, fd1)),
        );
        *out_f_c1.add(i) = F::scalar_add(
            fe[1],
            F::scalar_add(
                F::scalar_mul(challenge[0], fd1),
                F::scalar_mul(challenge[1], fd0),
            ),
        );

        let gd0 = F::scalar_sub(go_[0], ge[0]);
        let gd1 = F::scalar_sub(go_[1], ge[1]);
        *out_g_c0.add(i) = F::scalar_add(
            ge[0],
            F::scalar_add(F::scalar_mul(challenge[0], gd0), F::scalar_mul(ch1w_s, gd1)),
        );
        *out_g_c1.add(i) = F::scalar_add(
            ge[1],
            F::scalar_add(
                F::scalar_mul(challenge[0], gd1),
                F::scalar_mul(challenge[1], gd0),
            ),
        );

        i += 1;
    }

    (a, b)
}

/// Parallel fused SoA ext2 product reduce + evaluate.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn ext2_soa_product_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    use rayon::prelude::*;

    let n_out = out_f_c0.len();
    let chunk_pairs = 32_768_usize;
    if n_out <= chunk_pairs {
        return ext2_soa_product_reduce_and_evaluate_into::<F>(
            src_f_c0, src_f_c1, src_g_c0, src_g_c1, out_f_c0, out_f_c1, out_g_c0, out_g_c1,
            challenge, w,
        );
    }

    (out_f_c0.par_chunks_mut(chunk_pairs))
        .zip(out_f_c1.par_chunks_mut(chunk_pairs))
        .zip(out_g_c0.par_chunks_mut(chunk_pairs))
        .zip(out_g_c1.par_chunks_mut(chunk_pairs))
        .enumerate()
        .map(|(idx, (((ofc0, ofc1), ogc0), ogc1))| {
            let start = idx * chunk_pairs;
            let end = start + ofc0.len();
            ext2_soa_product_reduce_and_evaluate_into::<F>(
                &src_f_c0[2 * start..2 * end],
                &src_f_c1[2 * start..2 * end],
                &src_g_c0[2 * start..2 * end],
                &src_g_c1[2 * start..2 * end],
                ofc0,
                ofc1,
                ogc0,
                ogc1,
                challenge,
                w,
            )
        })
        .reduce(
            || ([0u64; 2], [0u64; 2]),
            |(a1, b1), (a2, b2)| {
                (
                    [F::scalar_add(a1[0], a2[0]), F::scalar_add(a1[1], a2[1])],
                    [F::scalar_add(b1[0], b2[0]), F::scalar_add(b1[1], b2[1])],
                )
            },
        )
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
pub fn ext2_soa_product_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    ext2_soa_product_reduce_and_evaluate_into::<F>(
        src_f_c0, src_f_c1, src_g_c0, src_g_c1, out_f_c0, out_f_c1, out_g_c0, out_g_c1, challenge,
        w,
    )
}

/// Fused SoA ext3 product evaluate + reduce in a single pass.
///
/// Same concept as ext2 fused product kernel but with Karatsuba ext3 multiply.
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_and_evaluate<F: SimdBaseField<Scalar = u64>>(
    f_c0: &mut [u64],
    f_c1: &mut [u64],
    f_c2: &mut [u64],
    g_c0: &mut [u64],
    g_c1: &mut [u64],
    g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3], usize) {
    let n = f_c0.len();
    let half = n / 2;

    // SAFETY: single-threaded ascending iteration is safe in-place.
    let (a, b) = unsafe {
        ext3_soa_product_reduce_and_evaluate_raw::<F>(
            f_c0.as_ptr(),
            f_c1.as_ptr(),
            f_c2.as_ptr(),
            g_c0.as_ptr(),
            g_c1.as_ptr(),
            g_c2.as_ptr(),
            f_c0.as_mut_ptr(),
            f_c1.as_mut_ptr(),
            f_c2.as_mut_ptr(),
            g_c0.as_mut_ptr(),
            g_c1.as_mut_ptr(),
            g_c2.as_mut_ptr(),
            half,
            challenge,
            w,
        )
    };
    (a, b, half)
}

/// Distinct-buffer version of `ext3_soa_product_reduce_and_evaluate`.
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_and_evaluate_into<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    let n_out = out_f_c0.len();
    debug_assert_eq!(src_f_c0.len(), 2 * n_out);
    unsafe {
        ext3_soa_product_reduce_and_evaluate_raw::<F>(
            src_f_c0.as_ptr(),
            src_f_c1.as_ptr(),
            src_f_c2.as_ptr(),
            src_g_c0.as_ptr(),
            src_g_c1.as_ptr(),
            src_g_c2.as_ptr(),
            out_f_c0.as_mut_ptr(),
            out_f_c1.as_mut_ptr(),
            out_f_c2.as_mut_ptr(),
            out_g_c0.as_mut_ptr(),
            out_g_c1.as_mut_ptr(),
            out_g_c2.as_mut_ptr(),
            n_out,
            challenge,
            w,
        )
    }
}

/// Raw-pointer core of `ext3_soa_product_reduce_and_evaluate`.
///
/// # Safety
/// Same contract as the ext2 product raw kernel.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ext3_soa_product_reduce_and_evaluate_raw<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: *const u64,
    src_f_c1: *const u64,
    src_f_c2: *const u64,
    src_g_c0: *const u64,
    src_g_c1: *const u64,
    src_g_c2: *const u64,
    out_f_c0: *mut u64,
    out_f_c1: *mut u64,
    out_f_c2: *mut u64,
    out_g_c0: *mut u64,
    out_g_c1: *mut u64,
    out_g_c2: *mut u64,
    n_out: usize,
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    let lanes = F::LANES;
    let aligned = (n_out / lanes) * lanes;
    let w_vec = F::splat(w);
    let ch = [
        F::splat(challenge[0]),
        F::splat(challenge[1]),
        F::splat(challenge[2]),
    ];

    let zero = F::splat(F::ZERO);
    let mut acc_a = [zero; 3];
    let mut acc_b = [zero; 3];

    let mut i = 0;
    while i < aligned {
        let off = i;
        let (fe0, fo0) = F::load_deinterleaved(src_f_c0.add(2 * off));
        let (fe1, fo1) = F::load_deinterleaved(src_f_c1.add(2 * off));
        let (fe2, fo2) = F::load_deinterleaved(src_f_c2.add(2 * off));
        let (ge0, go0) = F::load_deinterleaved(src_g_c0.add(2 * off));
        let (ge1, go1) = F::load_deinterleaved(src_g_c1.add(2 * off));
        let (ge2, go2) = F::load_deinterleaved(src_g_c2.add(2 * off));

        let pa = soa_ext3_mul::<F>([fe0, fe1, fe2], [ge0, ge1, ge2], w_vec);
        acc_a[0] = F::add(acc_a[0], pa[0]);
        acc_a[1] = F::add(acc_a[1], pa[1]);
        acc_a[2] = F::add(acc_a[2], pa[2]);

        let peg = soa_ext3_mul::<F>([fe0, fe1, fe2], [go0, go1, go2], w_vec);
        let poe = soa_ext3_mul::<F>([fo0, fo1, fo2], [ge0, ge1, ge2], w_vec);
        acc_b[0] = F::add(acc_b[0], F::add(peg[0], poe[0]));
        acc_b[1] = F::add(acc_b[1], F::add(peg[1], poe[1]));
        acc_b[2] = F::add(acc_b[2], F::add(peg[2], poe[2]));

        let fd = [F::sub(fo0, fe0), F::sub(fo1, fe1), F::sub(fo2, fe2)];
        let fp = soa_ext3_mul::<F>(ch, fd, w_vec);
        F::store(out_f_c0.add(off), F::add(fe0, fp[0]));
        F::store(out_f_c1.add(off), F::add(fe1, fp[1]));
        F::store(out_f_c2.add(off), F::add(fe2, fp[2]));

        let gd = [F::sub(go0, ge0), F::sub(go1, ge1), F::sub(go2, ge2)];
        let gp = soa_ext3_mul::<F>(ch, gd, w_vec);
        F::store(out_g_c0.add(off), F::add(ge0, gp[0]));
        F::store(out_g_c1.add(off), F::add(ge1, gp[1]));
        F::store(out_g_c2.add(off), F::add(ge2, gp[2]));
        i += lanes;
    }

    // Horizontal reduce
    let mut buf = [F::ZERO; 32];
    let mut a = [F::ZERO; 3];
    let mut b = [F::ZERO; 3];

    for c in 0..3 {
        F::store(buf.as_mut_ptr(), acc_a[c]);
        for &v in buf.iter().take(lanes) {
            a[c] = F::scalar_add(a[c], v);
        }
        F::store(buf.as_mut_ptr(), acc_b[c]);
        for &v in buf.iter().take(lanes) {
            b[c] = F::scalar_add(b[c], v);
        }
    }

    // Scalar tail
    while i < n_out {
        let fe = [
            *src_f_c0.add(2 * i),
            *src_f_c1.add(2 * i),
            *src_f_c2.add(2 * i),
        ];
        let fo = [
            *src_f_c0.add(2 * i + 1),
            *src_f_c1.add(2 * i + 1),
            *src_f_c2.add(2 * i + 1),
        ];
        let ge = [
            *src_g_c0.add(2 * i),
            *src_g_c1.add(2 * i),
            *src_g_c2.add(2 * i),
        ];
        let go_ = [
            *src_g_c0.add(2 * i + 1),
            *src_g_c1.add(2 * i + 1),
            *src_g_c2.add(2 * i + 1),
        ];

        let pa = scalar_ext3_mul::<F>(fe, ge, w);
        for c in 0..3 {
            a[c] = F::scalar_add(a[c], pa[c]);
        }

        let peg = scalar_ext3_mul::<F>(fe, go_, w);
        let poe = scalar_ext3_mul::<F>(fo, ge, w);
        for c in 0..3 {
            b[c] = F::scalar_add(b[c], F::scalar_add(peg[c], poe[c]));
        }

        let fd = [
            F::scalar_sub(fo[0], fe[0]),
            F::scalar_sub(fo[1], fe[1]),
            F::scalar_sub(fo[2], fe[2]),
        ];
        let fp = scalar_ext3_mul::<F>(challenge, fd, w);
        *out_f_c0.add(i) = F::scalar_add(fe[0], fp[0]);
        *out_f_c1.add(i) = F::scalar_add(fe[1], fp[1]);
        *out_f_c2.add(i) = F::scalar_add(fe[2], fp[2]);

        let gd = [
            F::scalar_sub(go_[0], ge[0]),
            F::scalar_sub(go_[1], ge[1]),
            F::scalar_sub(go_[2], ge[2]),
        ];
        let gp = scalar_ext3_mul::<F>(challenge, gd, w);
        *out_g_c0.add(i) = F::scalar_add(ge[0], gp[0]);
        *out_g_c1.add(i) = F::scalar_add(ge[1], gp[1]);
        *out_g_c2.add(i) = F::scalar_add(ge[2], gp[2]);

        i += 1;
    }

    (a, b)
}

/// Parallel fused SoA ext3 product reduce + evaluate.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    use rayon::prelude::*;

    let n_out = out_f_c0.len();
    let chunk_pairs = 32_768_usize;
    if n_out <= chunk_pairs {
        return ext3_soa_product_reduce_and_evaluate_into::<F>(
            src_f_c0, src_f_c1, src_f_c2, src_g_c0, src_g_c1, src_g_c2, out_f_c0, out_f_c1,
            out_f_c2, out_g_c0, out_g_c1, out_g_c2, challenge, w,
        );
    }

    // Zip six output-component slices by chunk index.
    (out_f_c0.par_chunks_mut(chunk_pairs))
        .zip(out_f_c1.par_chunks_mut(chunk_pairs))
        .zip(out_f_c2.par_chunks_mut(chunk_pairs))
        .zip(out_g_c0.par_chunks_mut(chunk_pairs))
        .zip(out_g_c1.par_chunks_mut(chunk_pairs))
        .zip(out_g_c2.par_chunks_mut(chunk_pairs))
        .enumerate()
        .map(|(idx, (((((ofc0, ofc1), ofc2), ogc0), ogc1), ogc2))| {
            let start = idx * chunk_pairs;
            let end = start + ofc0.len();
            ext3_soa_product_reduce_and_evaluate_into::<F>(
                &src_f_c0[2 * start..2 * end],
                &src_f_c1[2 * start..2 * end],
                &src_f_c2[2 * start..2 * end],
                &src_g_c0[2 * start..2 * end],
                &src_g_c1[2 * start..2 * end],
                &src_g_c2[2 * start..2 * end],
                ofc0,
                ofc1,
                ofc2,
                ogc0,
                ogc1,
                ogc2,
                challenge,
                w,
            )
        })
        .reduce(
            || ([0u64; 3], [0u64; 3]),
            |(a1, b1), (a2, b2)| {
                (
                    [
                        F::scalar_add(a1[0], a2[0]),
                        F::scalar_add(a1[1], a2[1]),
                        F::scalar_add(a1[2], a2[2]),
                    ],
                    [
                        F::scalar_add(b1[0], b2[0]),
                        F::scalar_add(b1[1], b2[1]),
                        F::scalar_add(b1[2], b2[2]),
                    ],
                )
            },
        )
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_and_evaluate_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    ext3_soa_product_reduce_and_evaluate_into::<F>(
        src_f_c0, src_f_c1, src_f_c2, src_g_c0, src_g_c1, src_g_c2, out_f_c0, out_f_c1, out_f_c2,
        out_g_c0, out_g_c1, out_g_c2, challenge, w,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Reduce-only kernels (no evaluate accumulation)
// ═══════════════════════════════════════════════════════════════════════════
//
// The fused reduce_and_evaluate kernels above also compute the current
// round's evaluate `(a, b) = (Σ f_e·g_e, Σ f_e·g_o + f_o·g_e)` as a side
// effect of the reduce pass. In practice every dispatch caller discards
// this return and recomputes `(a, b)` via the standalone
// `ext{2,3}_soa_product_evaluate` kernel at the top of each round — so the
// 3 extra ext{2,3} Karatsuba muls per iteration (pa, peg, poe) are pure
// waste. These `_reduce_only` variants skip that evaluate work.

// ── ext2 reduce-only ───────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ext2_soa_product_reduce_only_raw<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: *const u64,
    src_f_c1: *const u64,
    src_g_c0: *const u64,
    src_g_c1: *const u64,
    out_f_c0: *mut u64,
    out_f_c1: *mut u64,
    out_g_c0: *mut u64,
    out_g_c1: *mut u64,
    n_out: usize,
    challenge: [u64; 2],
    w: u64,
) {
    let lanes = F::LANES;
    let aligned = (n_out / lanes) * lanes;
    let ch0 = F::splat(challenge[0]);
    let ch1 = F::splat(challenge[1]);
    let ch1w = F::splat(F::scalar_mul(challenge[1], w));

    let mut i = 0;
    while i < aligned {
        let off = i;
        let (fe0, fo0) = F::load_deinterleaved(src_f_c0.add(2 * off));
        let (fe1, fo1) = F::load_deinterleaved(src_f_c1.add(2 * off));
        let (ge0, go0) = F::load_deinterleaved(src_g_c0.add(2 * off));
        let (ge1, go1) = F::load_deinterleaved(src_g_c1.add(2 * off));

        let fd0 = F::sub(fo0, fe0);
        let fd1 = F::sub(fo1, fe1);
        F::store(
            out_f_c0.add(off),
            F::add(fe0, F::add(F::mul(ch0, fd0), F::mul(ch1w, fd1))),
        );
        F::store(
            out_f_c1.add(off),
            F::add(fe1, F::add(F::mul(ch0, fd1), F::mul(ch1, fd0))),
        );

        let gd0 = F::sub(go0, ge0);
        let gd1 = F::sub(go1, ge1);
        F::store(
            out_g_c0.add(off),
            F::add(ge0, F::add(F::mul(ch0, gd0), F::mul(ch1w, gd1))),
        );
        F::store(
            out_g_c1.add(off),
            F::add(ge1, F::add(F::mul(ch0, gd1), F::mul(ch1, gd0))),
        );
        i += lanes;
    }

    let ch1w_s = F::scalar_mul(challenge[1], w);
    while i < n_out {
        let fe = [*src_f_c0.add(2 * i), *src_f_c1.add(2 * i)];
        let fo = [*src_f_c0.add(2 * i + 1), *src_f_c1.add(2 * i + 1)];
        let ge = [*src_g_c0.add(2 * i), *src_g_c1.add(2 * i)];
        let go_ = [*src_g_c0.add(2 * i + 1), *src_g_c1.add(2 * i + 1)];

        let fd0 = F::scalar_sub(fo[0], fe[0]);
        let fd1 = F::scalar_sub(fo[1], fe[1]);
        *out_f_c0.add(i) = F::scalar_add(
            fe[0],
            F::scalar_add(F::scalar_mul(challenge[0], fd0), F::scalar_mul(ch1w_s, fd1)),
        );
        *out_f_c1.add(i) = F::scalar_add(
            fe[1],
            F::scalar_add(
                F::scalar_mul(challenge[0], fd1),
                F::scalar_mul(challenge[1], fd0),
            ),
        );

        let gd0 = F::scalar_sub(go_[0], ge[0]);
        let gd1 = F::scalar_sub(go_[1], ge[1]);
        *out_g_c0.add(i) = F::scalar_add(
            ge[0],
            F::scalar_add(F::scalar_mul(challenge[0], gd0), F::scalar_mul(ch1w_s, gd1)),
        );
        *out_g_c1.add(i) = F::scalar_add(
            ge[1],
            F::scalar_add(
                F::scalar_mul(challenge[0], gd1),
                F::scalar_mul(challenge[1], gd0),
            ),
        );
        i += 1;
    }
}

/// In-place ext2 reduce (no evaluate). Returns the new length.
#[allow(clippy::too_many_arguments)]
pub fn ext2_soa_product_reduce_only<F: SimdBaseField<Scalar = u64>>(
    f_c0: &mut [u64],
    f_c1: &mut [u64],
    g_c0: &mut [u64],
    g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) -> usize {
    let n_elems = f_c0.len();
    debug_assert_eq!(n_elems, f_c1.len());
    debug_assert_eq!(n_elems, g_c0.len());
    debug_assert_eq!(n_elems, g_c1.len());
    let n_out = n_elems / 2;

    // SAFETY: all four slices have identical non-overlapping provenance.
    // The in-place reduce writes to `[0..n_out)` reading from `[0..n_elems)`;
    // writes to index i only ever read indices 2i and 2i+1, both ≥ i.
    unsafe {
        ext2_soa_product_reduce_only_raw::<F>(
            f_c0.as_ptr(),
            f_c1.as_ptr(),
            g_c0.as_ptr(),
            g_c1.as_ptr(),
            f_c0.as_mut_ptr(),
            f_c1.as_mut_ptr(),
            g_c0.as_mut_ptr(),
            g_c1.as_mut_ptr(),
            n_out,
            challenge,
            w,
        );
    }
    n_out
}

/// Distinct-buffer ext2 reduce (no evaluate).
#[allow(clippy::too_many_arguments)]
pub fn ext2_soa_product_reduce_only_into<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) {
    debug_assert_eq!(src_f_c0.len(), 2 * out_f_c0.len());
    let n_out = out_f_c0.len();
    unsafe {
        ext2_soa_product_reduce_only_raw::<F>(
            src_f_c0.as_ptr(),
            src_f_c1.as_ptr(),
            src_g_c0.as_ptr(),
            src_g_c1.as_ptr(),
            out_f_c0.as_mut_ptr(),
            out_f_c1.as_mut_ptr(),
            out_g_c0.as_mut_ptr(),
            out_g_c1.as_mut_ptr(),
            n_out,
            challenge,
            w,
        );
    }
}

/// Parallel ext2 reduce (no evaluate).
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn ext2_soa_product_reduce_only_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) {
    use rayon::prelude::*;

    let n_out = out_f_c0.len();
    let chunk_pairs = 32_768_usize;
    if n_out <= chunk_pairs {
        return ext2_soa_product_reduce_only_into::<F>(
            src_f_c0, src_f_c1, src_g_c0, src_g_c1, out_f_c0, out_f_c1, out_g_c0, out_g_c1,
            challenge, w,
        );
    }

    (out_f_c0.par_chunks_mut(chunk_pairs))
        .zip(out_f_c1.par_chunks_mut(chunk_pairs))
        .zip(out_g_c0.par_chunks_mut(chunk_pairs))
        .zip(out_g_c1.par_chunks_mut(chunk_pairs))
        .enumerate()
        .for_each(|(idx, (((ofc0, ofc1), ogc0), ogc1))| {
            let start = idx * chunk_pairs;
            let end = start + ofc0.len();
            ext2_soa_product_reduce_only_into::<F>(
                &src_f_c0[2 * start..2 * end],
                &src_f_c1[2 * start..2 * end],
                &src_g_c0[2 * start..2 * end],
                &src_g_c1[2 * start..2 * end],
                ofc0,
                ofc1,
                ogc0,
                ogc1,
                challenge,
                w,
            );
        });
}

#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
pub fn ext2_soa_product_reduce_only_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    challenge: [u64; 2],
    w: u64,
) {
    ext2_soa_product_reduce_only_into::<F>(
        src_f_c0, src_f_c1, src_g_c0, src_g_c1, out_f_c0, out_f_c1, out_g_c0, out_g_c1, challenge,
        w,
    )
}

// ── ext3 reduce-only ───────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ext3_soa_product_reduce_only_raw<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: *const u64,
    src_f_c1: *const u64,
    src_f_c2: *const u64,
    src_g_c0: *const u64,
    src_g_c1: *const u64,
    src_g_c2: *const u64,
    out_f_c0: *mut u64,
    out_f_c1: *mut u64,
    out_f_c2: *mut u64,
    out_g_c0: *mut u64,
    out_g_c1: *mut u64,
    out_g_c2: *mut u64,
    n_out: usize,
    challenge: [u64; 3],
    w: u64,
) {
    let lanes = F::LANES;
    let aligned = (n_out / lanes) * lanes;
    let w_vec = F::splat(w);
    let ch = [
        F::splat(challenge[0]),
        F::splat(challenge[1]),
        F::splat(challenge[2]),
    ];

    let mut i = 0;
    while i < aligned {
        let off = i;
        let (fe0, fo0) = F::load_deinterleaved(src_f_c0.add(2 * off));
        let (fe1, fo1) = F::load_deinterleaved(src_f_c1.add(2 * off));
        let (fe2, fo2) = F::load_deinterleaved(src_f_c2.add(2 * off));
        let (ge0, go0) = F::load_deinterleaved(src_g_c0.add(2 * off));
        let (ge1, go1) = F::load_deinterleaved(src_g_c1.add(2 * off));
        let (ge2, go2) = F::load_deinterleaved(src_g_c2.add(2 * off));

        let fd = [F::sub(fo0, fe0), F::sub(fo1, fe1), F::sub(fo2, fe2)];
        let fp = soa_ext3_mul::<F>(ch, fd, w_vec);
        F::store(out_f_c0.add(off), F::add(fe0, fp[0]));
        F::store(out_f_c1.add(off), F::add(fe1, fp[1]));
        F::store(out_f_c2.add(off), F::add(fe2, fp[2]));

        let gd = [F::sub(go0, ge0), F::sub(go1, ge1), F::sub(go2, ge2)];
        let gp = soa_ext3_mul::<F>(ch, gd, w_vec);
        F::store(out_g_c0.add(off), F::add(ge0, gp[0]));
        F::store(out_g_c1.add(off), F::add(ge1, gp[1]));
        F::store(out_g_c2.add(off), F::add(ge2, gp[2]));
        i += lanes;
    }

    // Scalar tail
    while i < n_out {
        let fe = [
            *src_f_c0.add(2 * i),
            *src_f_c1.add(2 * i),
            *src_f_c2.add(2 * i),
        ];
        let fo = [
            *src_f_c0.add(2 * i + 1),
            *src_f_c1.add(2 * i + 1),
            *src_f_c2.add(2 * i + 1),
        ];
        let ge = [
            *src_g_c0.add(2 * i),
            *src_g_c1.add(2 * i),
            *src_g_c2.add(2 * i),
        ];
        let go_ = [
            *src_g_c0.add(2 * i + 1),
            *src_g_c1.add(2 * i + 1),
            *src_g_c2.add(2 * i + 1),
        ];

        let fd = [
            F::scalar_sub(fo[0], fe[0]),
            F::scalar_sub(fo[1], fe[1]),
            F::scalar_sub(fo[2], fe[2]),
        ];
        let fp = scalar_ext3_mul::<F>(challenge, fd, w);
        *out_f_c0.add(i) = F::scalar_add(fe[0], fp[0]);
        *out_f_c1.add(i) = F::scalar_add(fe[1], fp[1]);
        *out_f_c2.add(i) = F::scalar_add(fe[2], fp[2]);

        let gd = [
            F::scalar_sub(go_[0], ge[0]),
            F::scalar_sub(go_[1], ge[1]),
            F::scalar_sub(go_[2], ge[2]),
        ];
        let gp = scalar_ext3_mul::<F>(challenge, gd, w);
        *out_g_c0.add(i) = F::scalar_add(ge[0], gp[0]);
        *out_g_c1.add(i) = F::scalar_add(ge[1], gp[1]);
        *out_g_c2.add(i) = F::scalar_add(ge[2], gp[2]);
        i += 1;
    }
}

/// In-place ext3 reduce (no evaluate). Returns the new length.
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_only<F: SimdBaseField<Scalar = u64>>(
    f_c0: &mut [u64],
    f_c1: &mut [u64],
    f_c2: &mut [u64],
    g_c0: &mut [u64],
    g_c1: &mut [u64],
    g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> usize {
    let n_elems = f_c0.len();
    let n_out = n_elems / 2;
    unsafe {
        ext3_soa_product_reduce_only_raw::<F>(
            f_c0.as_ptr(),
            f_c1.as_ptr(),
            f_c2.as_ptr(),
            g_c0.as_ptr(),
            g_c1.as_ptr(),
            g_c2.as_ptr(),
            f_c0.as_mut_ptr(),
            f_c1.as_mut_ptr(),
            f_c2.as_mut_ptr(),
            g_c0.as_mut_ptr(),
            g_c1.as_mut_ptr(),
            g_c2.as_mut_ptr(),
            n_out,
            challenge,
            w,
        );
    }
    n_out
}

/// Distinct-buffer ext3 reduce (no evaluate).
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_only_into<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) {
    let n_out = out_f_c0.len();
    unsafe {
        ext3_soa_product_reduce_only_raw::<F>(
            src_f_c0.as_ptr(),
            src_f_c1.as_ptr(),
            src_f_c2.as_ptr(),
            src_g_c0.as_ptr(),
            src_g_c1.as_ptr(),
            src_g_c2.as_ptr(),
            out_f_c0.as_mut_ptr(),
            out_f_c1.as_mut_ptr(),
            out_f_c2.as_mut_ptr(),
            out_g_c0.as_mut_ptr(),
            out_g_c1.as_mut_ptr(),
            out_g_c2.as_mut_ptr(),
            n_out,
            challenge,
            w,
        );
    }
}

/// Parallel ext3 reduce (no evaluate).
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_only_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) {
    use rayon::prelude::*;

    let n_out = out_f_c0.len();
    let chunk_pairs = 32_768_usize;
    if n_out <= chunk_pairs {
        return ext3_soa_product_reduce_only_into::<F>(
            src_f_c0, src_f_c1, src_f_c2, src_g_c0, src_g_c1, src_g_c2, out_f_c0, out_f_c1,
            out_f_c2, out_g_c0, out_g_c1, out_g_c2, challenge, w,
        );
    }

    (out_f_c0.par_chunks_mut(chunk_pairs))
        .zip(out_f_c1.par_chunks_mut(chunk_pairs))
        .zip(out_f_c2.par_chunks_mut(chunk_pairs))
        .zip(out_g_c0.par_chunks_mut(chunk_pairs))
        .zip(out_g_c1.par_chunks_mut(chunk_pairs))
        .zip(out_g_c2.par_chunks_mut(chunk_pairs))
        .enumerate()
        .for_each(|(idx, (((((ofc0, ofc1), ofc2), ogc0), ogc1), ogc2))| {
            let start = idx * chunk_pairs;
            let end = start + ofc0.len();
            ext3_soa_product_reduce_only_into::<F>(
                &src_f_c0[2 * start..2 * end],
                &src_f_c1[2 * start..2 * end],
                &src_f_c2[2 * start..2 * end],
                &src_g_c0[2 * start..2 * end],
                &src_g_c1[2 * start..2 * end],
                &src_g_c2[2 * start..2 * end],
                ofc0,
                ofc1,
                ofc2,
                ogc0,
                ogc1,
                ogc2,
                challenge,
                w,
            );
        });
}

#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_reduce_only_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) {
    ext3_soa_product_reduce_only_into::<F>(
        src_f_c0, src_f_c1, src_f_c2, src_g_c0, src_g_c1, src_g_c2, out_f_c0, out_f_c1, out_f_c2,
        out_g_c0, out_g_c1, out_g_c2, challenge, w,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Fused reduce + next-round-evaluate kernels
// ═══════════════════════════════════════════════════════════════════════════
//
// These kernels reduce the source (round k) into the output (round k+1)
// AND accumulate round k+1's evaluate `(a, b) = (Σ f_e·g_e, Σ f_e·g_o + f_o·g_e)`
// in a single pass over the data. Saves one full read of the source per
// round relative to the `reduce_only` + standalone `evaluate` pattern.
//
// Technique: 2× unroll the SIMD loop so each iteration stores `2·LANES`
// reduced outputs to contiguous memory. Then a `load_deinterleaved` read
// of that just-written region (hot in L1) gives the round-k+1 even/odd
// pair split directly: `evens = [r[0], r[2], …, r[2L−2]]`,
// `odds  = [r[1], r[3], …, r[2L−1]]`. Those pairs feed the evaluate
// accumulators via the same `soa_ext{2,3}_mul` Karatsuba the reduce uses.
//
// For use with the `pending_eval` pattern in the dispatch: only round 0
// calls standalone evaluate; rounds 1+ consume the `(a_{k+1}, b_{k+1})`
// returned by this kernel from round k.

// ── ext3 fused reduce + next-round evaluate ────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ext3_soa_product_fused_reduce_next_eval_raw<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: *const u64,
    src_f_c1: *const u64,
    src_f_c2: *const u64,
    src_g_c0: *const u64,
    src_g_c1: *const u64,
    src_g_c2: *const u64,
    out_f_c0: *mut u64,
    out_f_c1: *mut u64,
    out_f_c2: *mut u64,
    out_g_c0: *mut u64,
    out_g_c1: *mut u64,
    out_g_c2: *mut u64,
    n_out: usize,
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    let lanes = F::LANES;
    let step = 2 * lanes; // 2× unroll: each iter produces 2·LANES reduced outputs
    let aligned = (n_out / step) * step;
    let w_vec = F::splat(w);
    let ch = [
        F::splat(challenge[0]),
        F::splat(challenge[1]),
        F::splat(challenge[2]),
    ];

    let zero = F::splat(F::ZERO);
    let mut acc_a = [zero; 3];
    let mut acc_b = [zero; 3];

    let mut i = 0;
    while i < aligned {
        // Unroll A: reduce LANES source pairs into reduced outputs [i .. i+lanes).
        let (fc0_fe_a, fc0_fo_a) = F::load_deinterleaved(src_f_c0.add(2 * i));
        let (fc1_fe_a, fc1_fo_a) = F::load_deinterleaved(src_f_c1.add(2 * i));
        let (fc2_fe_a, fc2_fo_a) = F::load_deinterleaved(src_f_c2.add(2 * i));
        let (gc0_ge_a, gc0_go_a) = F::load_deinterleaved(src_g_c0.add(2 * i));
        let (gc1_ge_a, gc1_go_a) = F::load_deinterleaved(src_g_c1.add(2 * i));
        let (gc2_ge_a, gc2_go_a) = F::load_deinterleaved(src_g_c2.add(2 * i));

        let fd_a = [
            F::sub(fc0_fo_a, fc0_fe_a),
            F::sub(fc1_fo_a, fc1_fe_a),
            F::sub(fc2_fo_a, fc2_fe_a),
        ];
        let fp_a = soa_ext3_mul::<F>(ch, fd_a, w_vec);
        F::store(out_f_c0.add(i), F::add(fc0_fe_a, fp_a[0]));
        F::store(out_f_c1.add(i), F::add(fc1_fe_a, fp_a[1]));
        F::store(out_f_c2.add(i), F::add(fc2_fe_a, fp_a[2]));

        let gd_a = [
            F::sub(gc0_go_a, gc0_ge_a),
            F::sub(gc1_go_a, gc1_ge_a),
            F::sub(gc2_go_a, gc2_ge_a),
        ];
        let gp_a = soa_ext3_mul::<F>(ch, gd_a, w_vec);
        F::store(out_g_c0.add(i), F::add(gc0_ge_a, gp_a[0]));
        F::store(out_g_c1.add(i), F::add(gc1_ge_a, gp_a[1]));
        F::store(out_g_c2.add(i), F::add(gc2_ge_a, gp_a[2]));

        // Unroll B: second LANES of reduced outputs [i+lanes .. i+2·lanes).
        let off_b = i + lanes;
        let (fc0_fe_b, fc0_fo_b) = F::load_deinterleaved(src_f_c0.add(2 * off_b));
        let (fc1_fe_b, fc1_fo_b) = F::load_deinterleaved(src_f_c1.add(2 * off_b));
        let (fc2_fe_b, fc2_fo_b) = F::load_deinterleaved(src_f_c2.add(2 * off_b));
        let (gc0_ge_b, gc0_go_b) = F::load_deinterleaved(src_g_c0.add(2 * off_b));
        let (gc1_ge_b, gc1_go_b) = F::load_deinterleaved(src_g_c1.add(2 * off_b));
        let (gc2_ge_b, gc2_go_b) = F::load_deinterleaved(src_g_c2.add(2 * off_b));

        let fd_b = [
            F::sub(fc0_fo_b, fc0_fe_b),
            F::sub(fc1_fo_b, fc1_fe_b),
            F::sub(fc2_fo_b, fc2_fe_b),
        ];
        let fp_b = soa_ext3_mul::<F>(ch, fd_b, w_vec);
        F::store(out_f_c0.add(off_b), F::add(fc0_fe_b, fp_b[0]));
        F::store(out_f_c1.add(off_b), F::add(fc1_fe_b, fp_b[1]));
        F::store(out_f_c2.add(off_b), F::add(fc2_fe_b, fp_b[2]));

        let gd_b = [
            F::sub(gc0_go_b, gc0_ge_b),
            F::sub(gc1_go_b, gc1_ge_b),
            F::sub(gc2_go_b, gc2_ge_b),
        ];
        let gp_b = soa_ext3_mul::<F>(ch, gd_b, w_vec);
        F::store(out_g_c0.add(off_b), F::add(gc0_ge_b, gp_b[0]));
        F::store(out_g_c1.add(off_b), F::add(gc1_ge_b, gp_b[1]));
        F::store(out_g_c2.add(off_b), F::add(gc2_ge_b, gp_b[2]));

        // Next-round evaluate: reload the just-stored 2·LANES reduced
        // outputs via deinterleave. `load_deinterleaved(ptr)` reads
        // `2·LANES` u64s starting at ptr and returns
        //   evens = [r[0], r[2], …, r[2L−2]]  (first elements of each pair)
        //   odds  = [r[1], r[3], …, r[2L−1]]  (second elements of each pair)
        // Those are exactly the round-k+1 even/odd component lanes.
        let (fc0_e, fc0_o) = F::load_deinterleaved(out_f_c0.add(i));
        let (fc1_e, fc1_o) = F::load_deinterleaved(out_f_c1.add(i));
        let (fc2_e, fc2_o) = F::load_deinterleaved(out_f_c2.add(i));
        let (gc0_e, gc0_o) = F::load_deinterleaved(out_g_c0.add(i));
        let (gc1_e, gc1_o) = F::load_deinterleaved(out_g_c1.add(i));
        let (gc2_e, gc2_o) = F::load_deinterleaved(out_g_c2.add(i));

        let pa = soa_ext3_mul::<F>([fc0_e, fc1_e, fc2_e], [gc0_e, gc1_e, gc2_e], w_vec);
        acc_a[0] = F::add(acc_a[0], pa[0]);
        acc_a[1] = F::add(acc_a[1], pa[1]);
        acc_a[2] = F::add(acc_a[2], pa[2]);

        let peg = soa_ext3_mul::<F>([fc0_e, fc1_e, fc2_e], [gc0_o, gc1_o, gc2_o], w_vec);
        let poe = soa_ext3_mul::<F>([fc0_o, fc1_o, fc2_o], [gc0_e, gc1_e, gc2_e], w_vec);
        acc_b[0] = F::add(acc_b[0], F::add(peg[0], poe[0]));
        acc_b[1] = F::add(acc_b[1], F::add(peg[1], poe[1]));
        acc_b[2] = F::add(acc_b[2], F::add(peg[2], poe[2]));

        i += step;
    }

    // Horizontal-reduce SIMD accumulators into scalar (a, b).
    let mut buf = [F::ZERO; 32];
    let mut a = [F::ZERO; 3];
    let mut b = [F::ZERO; 3];
    for c in 0..3 {
        F::store(buf.as_mut_ptr(), acc_a[c]);
        for &v in buf.iter().take(lanes) {
            a[c] = F::scalar_add(a[c], v);
        }
        F::store(buf.as_mut_ptr(), acc_b[c]);
        for &v in buf.iter().take(lanes) {
            b[c] = F::scalar_add(b[c], v);
        }
    }

    // Scalar tail: reduce pairs of elements at a time, accumulating next-eval.
    while i + 1 < n_out {
        // Reduce element i
        let fe_i = [
            *src_f_c0.add(2 * i),
            *src_f_c1.add(2 * i),
            *src_f_c2.add(2 * i),
        ];
        let fo_i = [
            *src_f_c0.add(2 * i + 1),
            *src_f_c1.add(2 * i + 1),
            *src_f_c2.add(2 * i + 1),
        ];
        let ge_i = [
            *src_g_c0.add(2 * i),
            *src_g_c1.add(2 * i),
            *src_g_c2.add(2 * i),
        ];
        let go_i = [
            *src_g_c0.add(2 * i + 1),
            *src_g_c1.add(2 * i + 1),
            *src_g_c2.add(2 * i + 1),
        ];
        let fd = [
            F::scalar_sub(fo_i[0], fe_i[0]),
            F::scalar_sub(fo_i[1], fe_i[1]),
            F::scalar_sub(fo_i[2], fe_i[2]),
        ];
        let fp = scalar_ext3_mul::<F>(challenge, fd, w);
        let r_f_i = [
            F::scalar_add(fe_i[0], fp[0]),
            F::scalar_add(fe_i[1], fp[1]),
            F::scalar_add(fe_i[2], fp[2]),
        ];
        *out_f_c0.add(i) = r_f_i[0];
        *out_f_c1.add(i) = r_f_i[1];
        *out_f_c2.add(i) = r_f_i[2];
        let gd = [
            F::scalar_sub(go_i[0], ge_i[0]),
            F::scalar_sub(go_i[1], ge_i[1]),
            F::scalar_sub(go_i[2], ge_i[2]),
        ];
        let gp = scalar_ext3_mul::<F>(challenge, gd, w);
        let r_g_i = [
            F::scalar_add(ge_i[0], gp[0]),
            F::scalar_add(ge_i[1], gp[1]),
            F::scalar_add(ge_i[2], gp[2]),
        ];
        *out_g_c0.add(i) = r_g_i[0];
        *out_g_c1.add(i) = r_g_i[1];
        *out_g_c2.add(i) = r_g_i[2];

        // Reduce element i+1
        let j = i + 1;
        let fe_j = [
            *src_f_c0.add(2 * j),
            *src_f_c1.add(2 * j),
            *src_f_c2.add(2 * j),
        ];
        let fo_j = [
            *src_f_c0.add(2 * j + 1),
            *src_f_c1.add(2 * j + 1),
            *src_f_c2.add(2 * j + 1),
        ];
        let ge_j = [
            *src_g_c0.add(2 * j),
            *src_g_c1.add(2 * j),
            *src_g_c2.add(2 * j),
        ];
        let go_j = [
            *src_g_c0.add(2 * j + 1),
            *src_g_c1.add(2 * j + 1),
            *src_g_c2.add(2 * j + 1),
        ];
        let fd_j = [
            F::scalar_sub(fo_j[0], fe_j[0]),
            F::scalar_sub(fo_j[1], fe_j[1]),
            F::scalar_sub(fo_j[2], fe_j[2]),
        ];
        let fp_j = scalar_ext3_mul::<F>(challenge, fd_j, w);
        let r_f_j = [
            F::scalar_add(fe_j[0], fp_j[0]),
            F::scalar_add(fe_j[1], fp_j[1]),
            F::scalar_add(fe_j[2], fp_j[2]),
        ];
        *out_f_c0.add(j) = r_f_j[0];
        *out_f_c1.add(j) = r_f_j[1];
        *out_f_c2.add(j) = r_f_j[2];
        let gd_j = [
            F::scalar_sub(go_j[0], ge_j[0]),
            F::scalar_sub(go_j[1], ge_j[1]),
            F::scalar_sub(go_j[2], ge_j[2]),
        ];
        let gp_j = scalar_ext3_mul::<F>(challenge, gd_j, w);
        let r_g_j = [
            F::scalar_add(ge_j[0], gp_j[0]),
            F::scalar_add(ge_j[1], gp_j[1]),
            F::scalar_add(ge_j[2], gp_j[2]),
        ];
        *out_g_c0.add(j) = r_g_j[0];
        *out_g_c1.add(j) = r_g_j[1];
        *out_g_c2.add(j) = r_g_j[2];

        // Next-eval: pair (r_f_i, r_f_j) × (r_g_i, r_g_j)
        let pa = scalar_ext3_mul::<F>(r_f_i, r_g_i, w);
        a[0] = F::scalar_add(a[0], pa[0]);
        a[1] = F::scalar_add(a[1], pa[1]);
        a[2] = F::scalar_add(a[2], pa[2]);
        let peg = scalar_ext3_mul::<F>(r_f_i, r_g_j, w);
        let poe = scalar_ext3_mul::<F>(r_f_j, r_g_i, w);
        b[0] = F::scalar_add(b[0], F::scalar_add(peg[0], poe[0]));
        b[1] = F::scalar_add(b[1], F::scalar_add(peg[1], poe[1]));
        b[2] = F::scalar_add(b[2], F::scalar_add(peg[2], poe[2]));

        i += 2;
    }

    // Final straggler: if n_out is odd, reduce the last element without
    // contributing to next-round-eval (no pair to form).
    if i < n_out {
        let fe = [
            *src_f_c0.add(2 * i),
            *src_f_c1.add(2 * i),
            *src_f_c2.add(2 * i),
        ];
        let fo = [
            *src_f_c0.add(2 * i + 1),
            *src_f_c1.add(2 * i + 1),
            *src_f_c2.add(2 * i + 1),
        ];
        let ge = [
            *src_g_c0.add(2 * i),
            *src_g_c1.add(2 * i),
            *src_g_c2.add(2 * i),
        ];
        let go = [
            *src_g_c0.add(2 * i + 1),
            *src_g_c1.add(2 * i + 1),
            *src_g_c2.add(2 * i + 1),
        ];
        let fd = [
            F::scalar_sub(fo[0], fe[0]),
            F::scalar_sub(fo[1], fe[1]),
            F::scalar_sub(fo[2], fe[2]),
        ];
        let fp = scalar_ext3_mul::<F>(challenge, fd, w);
        *out_f_c0.add(i) = F::scalar_add(fe[0], fp[0]);
        *out_f_c1.add(i) = F::scalar_add(fe[1], fp[1]);
        *out_f_c2.add(i) = F::scalar_add(fe[2], fp[2]);
        let gd = [
            F::scalar_sub(go[0], ge[0]),
            F::scalar_sub(go[1], ge[1]),
            F::scalar_sub(go[2], ge[2]),
        ];
        let gp = scalar_ext3_mul::<F>(challenge, gd, w);
        *out_g_c0.add(i) = F::scalar_add(ge[0], gp[0]);
        *out_g_c1.add(i) = F::scalar_add(ge[1], gp[1]);
        *out_g_c2.add(i) = F::scalar_add(ge[2], gp[2]);
    }

    (a, b)
}

/// In-place ext3 fused reduce + next-round evaluate. Writes reduced output
/// to the first half of each input slice. Returns `(a_{k+1}, b_{k+1})` — the
/// round-k+1 evaluate coefficients computed from the just-reduced output.
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_fused_reduce_next_eval<F: SimdBaseField<Scalar = u64>>(
    f_c0: &mut [u64],
    f_c1: &mut [u64],
    f_c2: &mut [u64],
    g_c0: &mut [u64],
    g_c1: &mut [u64],
    g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3], usize) {
    let n_elems = f_c0.len();
    let n_out = n_elems / 2;
    // SAFETY: same aliasing reasoning as `ext3_soa_product_reduce_only`: each
    // iteration reads src[2i], src[2i+1] and writes out[i] where 2i ≥ i, so
    // in-place mutation never clobbers unread source. The fused next-eval
    // reload reads out[i..i+2·LANES], which were both just written in the
    // current iteration.
    let (a, b) = unsafe {
        ext3_soa_product_fused_reduce_next_eval_raw::<F>(
            f_c0.as_ptr(),
            f_c1.as_ptr(),
            f_c2.as_ptr(),
            g_c0.as_ptr(),
            g_c1.as_ptr(),
            g_c2.as_ptr(),
            f_c0.as_mut_ptr(),
            f_c1.as_mut_ptr(),
            f_c2.as_mut_ptr(),
            g_c0.as_mut_ptr(),
            g_c1.as_mut_ptr(),
            g_c2.as_mut_ptr(),
            n_out,
            challenge,
            w,
        )
    };
    (a, b, n_out)
}

/// Distinct-buffer ext3 fused reduce + next-round evaluate. Returns
/// `(a_{k+1}, b_{k+1})` — the round-k+1 evaluate coefficients computed
/// from the just-reduced output.
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_fused_reduce_next_eval_into<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    let n_out = out_f_c0.len();
    unsafe {
        ext3_soa_product_fused_reduce_next_eval_raw::<F>(
            src_f_c0.as_ptr(),
            src_f_c1.as_ptr(),
            src_f_c2.as_ptr(),
            src_g_c0.as_ptr(),
            src_g_c1.as_ptr(),
            src_g_c2.as_ptr(),
            out_f_c0.as_mut_ptr(),
            out_f_c1.as_mut_ptr(),
            out_f_c2.as_mut_ptr(),
            out_g_c0.as_mut_ptr(),
            out_g_c1.as_mut_ptr(),
            out_g_c2.as_mut_ptr(),
            n_out,
            challenge,
            w,
        )
    }
}

/// Parallel ext3 fused reduce + next-round evaluate.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_fused_reduce_next_eval_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    use rayon::prelude::*;

    let n_out = out_f_c0.len();
    let chunk_pairs = 32_768_usize;
    if n_out <= chunk_pairs {
        return ext3_soa_product_fused_reduce_next_eval_into::<F>(
            src_f_c0, src_f_c1, src_f_c2, src_g_c0, src_g_c1, src_g_c2, out_f_c0, out_f_c1,
            out_f_c2, out_g_c0, out_g_c1, out_g_c2, challenge, w,
        );
    }

    (out_f_c0.par_chunks_mut(chunk_pairs))
        .zip(out_f_c1.par_chunks_mut(chunk_pairs))
        .zip(out_f_c2.par_chunks_mut(chunk_pairs))
        .zip(out_g_c0.par_chunks_mut(chunk_pairs))
        .zip(out_g_c1.par_chunks_mut(chunk_pairs))
        .zip(out_g_c2.par_chunks_mut(chunk_pairs))
        .enumerate()
        .map(|(idx, (((((ofc0, ofc1), ofc2), ogc0), ogc1), ogc2))| {
            let start = idx * chunk_pairs;
            let end = start + ofc0.len();
            ext3_soa_product_fused_reduce_next_eval_into::<F>(
                &src_f_c0[2 * start..2 * end],
                &src_f_c1[2 * start..2 * end],
                &src_f_c2[2 * start..2 * end],
                &src_g_c0[2 * start..2 * end],
                &src_g_c1[2 * start..2 * end],
                &src_g_c2[2 * start..2 * end],
                ofc0,
                ofc1,
                ofc2,
                ogc0,
                ogc1,
                ogc2,
                challenge,
                w,
            )
        })
        .reduce(
            || ([0u64; 3], [0u64; 3]),
            |(a1, b1), (a2, b2)| {
                (
                    [
                        F::scalar_add(a1[0], a2[0]),
                        F::scalar_add(a1[1], a2[1]),
                        F::scalar_add(a1[2], a2[2]),
                    ],
                    [
                        F::scalar_add(b1[0], b2[0]),
                        F::scalar_add(b1[1], b2[1]),
                        F::scalar_add(b1[2], b2[2]),
                    ],
                )
            },
        )
}

#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
pub fn ext3_soa_product_fused_reduce_next_eval_parallel<F: SimdBaseField<Scalar = u64>>(
    src_f_c0: &[u64],
    src_f_c1: &[u64],
    src_f_c2: &[u64],
    src_g_c0: &[u64],
    src_g_c1: &[u64],
    src_g_c2: &[u64],
    out_f_c0: &mut [u64],
    out_f_c1: &mut [u64],
    out_f_c2: &mut [u64],
    out_g_c0: &mut [u64],
    out_g_c1: &mut [u64],
    out_g_c2: &mut [u64],
    challenge: [u64; 3],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    ext3_soa_product_fused_reduce_next_eval_into::<F>(
        src_f_c0, src_f_c1, src_f_c2, src_g_c0, src_g_c1, src_g_c2, out_f_c0, out_f_c1, out_f_c2,
        out_g_c0, out_g_c1, out_g_c2, challenge, w,
    )
}

/// SoA ext2 inner product evaluate.
///
/// Given `f` and `g` as ext2 elements in SoA layout (f_c0, f_c1, g_c0, g_c1),
/// computes the degree-2 round polynomial coefficients `(a, b)`:
///   a = Σ f[2i] * g[2i]                        (ext2 products)
///   b = Σ (f[2i] * g[2i+1] + f[2i+1] * g[2i])  (ext2 cross-terms)
///
/// Returns `(a_c0, a_c1, b_c0, b_c1)` as raw u64 components.
pub fn ext2_soa_product_evaluate<F: SimdBaseField<Scalar = u64>>(
    f_c0: &[u64],
    f_c1: &[u64],
    g_c0: &[u64],
    g_c1: &[u64],
    w: u64,
) -> ([u64; 2], [u64; 2]) {
    let n = f_c0.len();
    debug_assert_eq!(n, f_c1.len());
    debug_assert_eq!(n, g_c0.len());
    debug_assert_eq!(n, g_c1.len());

    let lanes = F::LANES;
    // Each load_deinterleaved consumes 2*lanes u64s of input (covering `lanes` pairs).
    // 2× unroll → each iteration consumes 4*lanes u64s.
    let load_width = 2 * lanes;
    let step = 2 * load_width; // 4 * lanes
    let aligned = (n / step) * step;
    let w_vec = F::splat(w);

    let zero = F::splat(F::ZERO);
    let mut acc_a0 = zero;
    let mut acc_a1 = zero;
    let mut acc_b0 = zero;
    let mut acc_b1 = zero;

    let mut i = 0;
    while i < aligned {
        unsafe {
            for u in 0..2 {
                let off = i + u * load_width;
                let (fe0, fo0) = F::load_deinterleaved(f_c0.as_ptr().add(off));
                let (fe1, fo1) = F::load_deinterleaved(f_c1.as_ptr().add(off));
                let (ge0, go0) = F::load_deinterleaved(g_c0.as_ptr().add(off));
                let (ge1, go1) = F::load_deinterleaved(g_c1.as_ptr().add(off));

                // a += f_even * g_even (ext2 Karatsuba)
                let v0 = F::mul(fe0, ge0);
                let v1 = F::mul(fe1, ge1);
                acc_a0 = F::add(acc_a0, F::add(v0, F::mul(w_vec, v1)));
                let m = F::mul(F::add(fe0, fe1), F::add(ge0, ge1));
                acc_a1 = F::add(acc_a1, F::sub(F::sub(m, v0), v1));

                // b += f_even * g_odd (ext2 Karatsuba)
                let u0 = F::mul(fe0, go0);
                let u1 = F::mul(fe1, go1);
                let m1 = F::mul(F::add(fe0, fe1), F::add(go0, go1));
                // b += f_odd * g_even (ext2 Karatsuba)
                let p0 = F::mul(fo0, ge0);
                let p1 = F::mul(fo1, ge1);
                let m2 = F::mul(F::add(fo0, fo1), F::add(ge0, ge1));

                acc_b0 = F::add(
                    acc_b0,
                    F::add(F::add(u0, F::mul(w_vec, u1)), F::add(p0, F::mul(w_vec, p1))),
                );
                acc_b1 = F::add(
                    acc_b1,
                    F::add(F::sub(F::sub(m1, u0), u1), F::sub(F::sub(m2, p0), p1)),
                );
            }
        }
        i += step;
    }

    // Remaining SIMD vectors (one load_width at a time)
    while i + load_width <= n {
        unsafe {
            let (fe0, fo0) = F::load_deinterleaved(f_c0.as_ptr().add(i));
            let (fe1, fo1) = F::load_deinterleaved(f_c1.as_ptr().add(i));
            let (ge0, go0) = F::load_deinterleaved(g_c0.as_ptr().add(i));
            let (ge1, go1) = F::load_deinterleaved(g_c1.as_ptr().add(i));

            let v0 = F::mul(fe0, ge0);
            let v1 = F::mul(fe1, ge1);
            acc_a0 = F::add(acc_a0, F::add(v0, F::mul(w_vec, v1)));
            let m = F::mul(F::add(fe0, fe1), F::add(ge0, ge1));
            acc_a1 = F::add(acc_a1, F::sub(F::sub(m, v0), v1));

            let u0 = F::mul(fe0, go0);
            let u1 = F::mul(fe1, go1);
            let m1 = F::mul(F::add(fe0, fe1), F::add(go0, go1));
            let p0 = F::mul(fo0, ge0);
            let p1 = F::mul(fo1, ge1);
            let m2 = F::mul(F::add(fo0, fo1), F::add(ge0, ge1));

            acc_b0 = F::add(
                acc_b0,
                F::add(F::add(u0, F::mul(w_vec, u1)), F::add(p0, F::mul(w_vec, p1))),
            );
            acc_b1 = F::add(
                acc_b1,
                F::add(F::sub(F::sub(m1, u0), u1), F::sub(F::sub(m2, p0), p1)),
            );
        }
        i += load_width;
    }

    // Horizontal reduce
    let mut buf = [F::ZERO; 32];
    let mut a = [F::ZERO; 2];
    let mut b = [F::ZERO; 2];

    unsafe { F::store(buf.as_mut_ptr(), acc_a0) };
    for &v in buf.iter().take(lanes) {
        a[0] = F::scalar_add(a[0], v);
    }
    unsafe { F::store(buf.as_mut_ptr(), acc_a1) };
    for &v in buf.iter().take(lanes) {
        a[1] = F::scalar_add(a[1], v);
    }
    unsafe { F::store(buf.as_mut_ptr(), acc_b0) };
    for &v in buf.iter().take(lanes) {
        b[0] = F::scalar_add(b[0], v);
    }
    unsafe { F::store(buf.as_mut_ptr(), acc_b1) };
    for &v in buf.iter().take(lanes) {
        b[1] = F::scalar_add(b[1], v);
    }

    // Scalar tail
    while i + 1 < n {
        let fe = [f_c0[i], f_c1[i]];
        let fo = [f_c0[i + 1], f_c1[i + 1]];
        let ge = [g_c0[i], g_c1[i]];
        let go_ = [g_c0[i + 1], g_c1[i + 1]];

        // a += fe * ge
        let v0 = F::scalar_mul(fe[0], ge[0]);
        let v1 = F::scalar_mul(fe[1], ge[1]);
        a[0] = F::scalar_add(a[0], F::scalar_add(v0, F::scalar_mul(w, v1)));
        let m = F::scalar_mul(F::scalar_add(fe[0], fe[1]), F::scalar_add(ge[0], ge[1]));
        a[1] = F::scalar_add(a[1], F::scalar_sub(F::scalar_sub(m, v0), v1));

        // b += fe * go + fo * ge
        let u0 = F::scalar_mul(fe[0], go_[0]);
        let u1 = F::scalar_mul(fe[1], go_[1]);
        let m1 = F::scalar_mul(F::scalar_add(fe[0], fe[1]), F::scalar_add(go_[0], go_[1]));
        let p0 = F::scalar_mul(fo[0], ge[0]);
        let p1 = F::scalar_mul(fo[1], ge[1]);
        let m2 = F::scalar_mul(F::scalar_add(fo[0], fo[1]), F::scalar_add(ge[0], ge[1]));

        b[0] = F::scalar_add(
            b[0],
            F::scalar_add(
                F::scalar_add(u0, F::scalar_mul(w, u1)),
                F::scalar_add(p0, F::scalar_mul(w, p1)),
            ),
        );
        b[1] = F::scalar_add(
            b[1],
            F::scalar_add(
                F::scalar_sub(F::scalar_sub(m1, u0), u1),
                F::scalar_sub(F::scalar_sub(m2, p0), p1),
            ),
        );
        i += 2;
    }

    (a, b)
}

/// SoA ext3 inner product evaluate.
///
/// Given `f` and `g` as ext3 elements in SoA layout (f_c0, f_c1, f_c2, g_c0, g_c1, g_c2),
/// computes the degree-2 round polynomial coefficients `(a, b)`:
///   a = Σ f[2i] * g[2i]                        (ext3 products)
///   b = Σ (f[2i] * g[2i+1] + f[2i+1] * g[2i])  (ext3 cross-terms)
///
/// Returns `(a_components, b_components)` as `[u64; 3]` raw Montgomery values.
pub fn ext3_soa_product_evaluate<F: SimdBaseField<Scalar = u64>>(
    f_c0: &[u64],
    f_c1: &[u64],
    f_c2: &[u64],
    g_c0: &[u64],
    g_c1: &[u64],
    g_c2: &[u64],
    w: u64,
) -> ([u64; 3], [u64; 3]) {
    let n = f_c0.len();
    debug_assert_eq!(n, f_c1.len());
    debug_assert_eq!(n, f_c2.len());
    debug_assert_eq!(n, g_c0.len());
    debug_assert_eq!(n, g_c1.len());
    debug_assert_eq!(n, g_c2.len());

    let lanes = F::LANES;
    // Each load_deinterleaved consumes 2*lanes u64s (one load_width). 2× unroll.
    let load_width = 2 * lanes;
    let step = 2 * load_width; // 4 * lanes
    let aligned = (n / step) * step;
    let w_vec = F::splat(w);

    let zero = F::splat(F::ZERO);
    // Accumulators for a (3 components) and b (3 components)
    let mut acc_a = [zero; 3];
    let mut acc_b = [zero; 3];

    let mut i = 0;
    while i < aligned {
        unsafe {
            for u in 0..2 {
                let off = i + u * load_width;
                let (fe0, fo0) = F::load_deinterleaved(f_c0.as_ptr().add(off));
                let (fe1, fo1) = F::load_deinterleaved(f_c1.as_ptr().add(off));
                let (fe2, fo2) = F::load_deinterleaved(f_c2.as_ptr().add(off));
                let (ge0, go0) = F::load_deinterleaved(g_c0.as_ptr().add(off));
                let (ge1, go1) = F::load_deinterleaved(g_c1.as_ptr().add(off));
                let (ge2, go2) = F::load_deinterleaved(g_c2.as_ptr().add(off));

                // a += f_even * g_even (ext3 Karatsuba)
                let prod_a = soa_ext3_mul::<F>([fe0, fe1, fe2], [ge0, ge1, ge2], w_vec);
                acc_a[0] = F::add(acc_a[0], prod_a[0]);
                acc_a[1] = F::add(acc_a[1], prod_a[1]);
                acc_a[2] = F::add(acc_a[2], prod_a[2]);

                // b += f_even * g_odd + f_odd * g_even
                let prod_eg = soa_ext3_mul::<F>([fe0, fe1, fe2], [go0, go1, go2], w_vec);
                let prod_oe = soa_ext3_mul::<F>([fo0, fo1, fo2], [ge0, ge1, ge2], w_vec);
                acc_b[0] = F::add(acc_b[0], F::add(prod_eg[0], prod_oe[0]));
                acc_b[1] = F::add(acc_b[1], F::add(prod_eg[1], prod_oe[1]));
                acc_b[2] = F::add(acc_b[2], F::add(prod_eg[2], prod_oe[2]));
            }
        }
        i += step;
    }

    // Remaining SIMD vectors (one load_width at a time)
    while i + load_width <= n {
        unsafe {
            let (fe0, fo0) = F::load_deinterleaved(f_c0.as_ptr().add(i));
            let (fe1, fo1) = F::load_deinterleaved(f_c1.as_ptr().add(i));
            let (fe2, fo2) = F::load_deinterleaved(f_c2.as_ptr().add(i));
            let (ge0, go0) = F::load_deinterleaved(g_c0.as_ptr().add(i));
            let (ge1, go1) = F::load_deinterleaved(g_c1.as_ptr().add(i));
            let (ge2, go2) = F::load_deinterleaved(g_c2.as_ptr().add(i));

            let prod_a = soa_ext3_mul::<F>([fe0, fe1, fe2], [ge0, ge1, ge2], w_vec);
            acc_a[0] = F::add(acc_a[0], prod_a[0]);
            acc_a[1] = F::add(acc_a[1], prod_a[1]);
            acc_a[2] = F::add(acc_a[2], prod_a[2]);

            let prod_eg = soa_ext3_mul::<F>([fe0, fe1, fe2], [go0, go1, go2], w_vec);
            let prod_oe = soa_ext3_mul::<F>([fo0, fo1, fo2], [ge0, ge1, ge2], w_vec);
            acc_b[0] = F::add(acc_b[0], F::add(prod_eg[0], prod_oe[0]));
            acc_b[1] = F::add(acc_b[1], F::add(prod_eg[1], prod_oe[1]));
            acc_b[2] = F::add(acc_b[2], F::add(prod_eg[2], prod_oe[2]));
        }
        i += load_width;
    }

    // Horizontal reduce
    let mut buf = [F::ZERO; 32];
    let mut a = [F::ZERO; 3];
    let mut b = [F::ZERO; 3];

    for c in 0..3 {
        unsafe { F::store(buf.as_mut_ptr(), acc_a[c]) };
        for &v in buf.iter().take(lanes) {
            a[c] = F::scalar_add(a[c], v);
        }
        unsafe { F::store(buf.as_mut_ptr(), acc_b[c]) };
        for &v in buf.iter().take(lanes) {
            b[c] = F::scalar_add(b[c], v);
        }
    }

    // Scalar tail
    while i + 1 < n {
        let fe = [f_c0[i], f_c1[i], f_c2[i]];
        let fo = [f_c0[i + 1], f_c1[i + 1], f_c2[i + 1]];
        let ge = [g_c0[i], g_c1[i], g_c2[i]];
        let go_ = [g_c0[i + 1], g_c1[i + 1], g_c2[i + 1]];

        let pa = scalar_ext3_mul::<F>(fe, ge, w);
        for c in 0..3 {
            a[c] = F::scalar_add(a[c], pa[c]);
        }

        let peg = scalar_ext3_mul::<F>(fe, go_, w);
        let poe = scalar_ext3_mul::<F>(fo, ge, w);
        for c in 0..3 {
            b[c] = F::scalar_add(b[c], F::scalar_add(peg[c], poe[c]));
        }

        i += 2;
    }

    (a, b)
}

/// Ext3 Karatsuba multiply for SIMD vectors in SoA layout.
/// 6 base muls + 2 w-muls + adds.
#[inline(always)]
fn soa_ext3_mul<F: SimdBaseField<Scalar = u64>>(
    a: [F::Packed; 3],
    b: [F::Packed; 3],
    w: F::Packed,
) -> [F::Packed; 3] {
    let ad = F::mul(a[0], b[0]);
    let be = F::mul(a[1], b[1]);
    let cf = F::mul(a[2], b[2]);

    let x = F::sub(
        F::sub(F::mul(F::add(a[1], a[2]), F::add(b[1], b[2])), be),
        cf,
    );
    let y = F::sub(
        F::sub(F::mul(F::add(a[0], a[1]), F::add(b[0], b[1])), ad),
        be,
    );
    let z = F::add(
        F::sub(
            F::sub(F::mul(F::add(a[0], a[2]), F::add(b[0], b[2])), ad),
            cf,
        ),
        be,
    );

    [F::add(ad, F::mul(w, x)), F::add(y, F::mul(w, cf)), z]
}

/// Scalar ext3 Karatsuba multiply helper.
#[inline(always)]
fn scalar_ext3_mul<F: SimdBaseField<Scalar = u64>>(a: [u64; 3], b: [u64; 3], w: u64) -> [u64; 3] {
    let ad = F::scalar_mul(a[0], b[0]);
    let be = F::scalar_mul(a[1], b[1]);
    let cf = F::scalar_mul(a[2], b[2]);

    let x = F::scalar_sub(
        F::scalar_sub(
            F::scalar_mul(F::scalar_add(a[1], a[2]), F::scalar_add(b[1], b[2])),
            be,
        ),
        cf,
    );
    let y = F::scalar_sub(
        F::scalar_sub(
            F::scalar_mul(F::scalar_add(a[0], a[1]), F::scalar_add(b[0], b[1])),
            ad,
        ),
        be,
    );
    let z = F::scalar_add(
        F::scalar_sub(
            F::scalar_sub(
                F::scalar_mul(F::scalar_add(a[0], a[2]), F::scalar_add(b[0], b[2])),
                ad,
            ),
            cf,
        ),
        be,
    );

    [
        F::scalar_add(ad, F::scalar_mul(w, x)),
        F::scalar_add(y, F::scalar_mul(w, cf)),
        z,
    ]
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

    /// Verify the ext3 fused reduce+next-eval kernel produces the same
    /// (reduced_data, a_{k+1}, b_{k+1}) as `reduce_only` followed by a
    /// standalone `ext3_soa_product_evaluate` on the reduced data.
    #[test]
    fn test_ext3_fused_reduce_next_eval_matches_separate() {
        use crate::tests::F64Ext3;
        use ark_ff::UniformRand;

        let mut rng = test_rng();
        // Exercise SIMD main loop (n_out ≥ 2·LANES), scalar-tail pair, and
        // straggler paths: large even, large odd, small.
        for num_source_pairs in [32usize, 31, 17, 16, 8, 4, 2] {
            let n_src = 2 * num_source_pairs;
            // Sample random ext3 sources and convert to SoA u64 columns.
            let f: Vec<F64Ext3> = (0..n_src).map(|_| F64Ext3::rand(&mut rng)).collect();
            let g: Vec<F64Ext3> = (0..n_src).map(|_| F64Ext3::rand(&mut rng)).collect();
            let (f_c0, f_c1, f_c2): (Vec<u64>, Vec<u64>, Vec<u64>) = {
                let mut c0 = Vec::with_capacity(n_src);
                let mut c1 = Vec::with_capacity(n_src);
                let mut c2 = Vec::with_capacity(n_src);
                for x in &f {
                    let bytes: [u64; 3] = unsafe { *(x as *const F64Ext3 as *const [u64; 3]) };
                    c0.push(bytes[0]);
                    c1.push(bytes[1]);
                    c2.push(bytes[2]);
                }
                (c0, c1, c2)
            };
            let (g_c0, g_c1, g_c2): (Vec<u64>, Vec<u64>, Vec<u64>) = {
                let mut c0 = Vec::with_capacity(n_src);
                let mut c1 = Vec::with_capacity(n_src);
                let mut c2 = Vec::with_capacity(n_src);
                for x in &g {
                    let bytes: [u64; 3] = unsafe { *(x as *const F64Ext3 as *const [u64; 3]) };
                    c0.push(bytes[0]);
                    c1.push(bytes[1]);
                    c2.push(bytes[2]);
                }
                (c0, c1, c2)
            };

            // Random challenge and nonresidue in Montgomery form — use a
            // small concrete nonresidue; the kernel treats `w` as opaque.
            let chg: [u64; 3] = {
                let c = F64Ext3::rand(&mut rng);
                unsafe { *(&c as *const F64Ext3 as *const [u64; 3]) }
            };
            let w: u64 = {
                let nr = F64Ext3::rand(&mut rng);
                unsafe { *(&nr as *const F64Ext3 as *const u64) }
            };

            // Reference: reduce_only then standalone evaluate on reduced.
            let mut ref_out_f = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            let mut ref_out_g = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            ext3_soa_product_reduce_only_into::<Backend>(
                &f_c0,
                &f_c1,
                &f_c2,
                &g_c0,
                &g_c1,
                &g_c2,
                &mut ref_out_f.0,
                &mut ref_out_f.1,
                &mut ref_out_f.2,
                &mut ref_out_g.0,
                &mut ref_out_g.1,
                &mut ref_out_g.2,
                chg,
                w,
            );
            // Next-round evaluate: only defined when n_out ≥ 2.
            let (ref_a, ref_b) = if n_src / 2 >= 2 {
                ext3_soa_product_evaluate::<Backend>(
                    &ref_out_f.0,
                    &ref_out_f.1,
                    &ref_out_f.2,
                    &ref_out_g.0,
                    &ref_out_g.1,
                    &ref_out_g.2,
                    w,
                )
            } else {
                ([0u64; 3], [0u64; 3])
            };

            // Under test: fused kernel.
            let mut got_out_f = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            let mut got_out_g = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            let (got_a, got_b) = ext3_soa_product_fused_reduce_next_eval_into::<Backend>(
                &f_c0,
                &f_c1,
                &f_c2,
                &g_c0,
                &g_c1,
                &g_c2,
                &mut got_out_f.0,
                &mut got_out_f.1,
                &mut got_out_f.2,
                &mut got_out_g.0,
                &mut got_out_g.1,
                &mut got_out_g.2,
                chg,
                w,
            );

            assert_eq!(got_out_f.0, ref_out_f.0, "f_c0 mismatch (n_src={})", n_src);
            assert_eq!(got_out_f.1, ref_out_f.1, "f_c1 mismatch (n_src={})", n_src);
            assert_eq!(got_out_f.2, ref_out_f.2, "f_c2 mismatch (n_src={})", n_src);
            assert_eq!(got_out_g.0, ref_out_g.0, "g_c0 mismatch (n_src={})", n_src);
            assert_eq!(got_out_g.1, ref_out_g.1, "g_c1 mismatch (n_src={})", n_src);
            assert_eq!(got_out_g.2, ref_out_g.2, "g_c2 mismatch (n_src={})", n_src);
            if n_src / 2 >= 2 {
                assert_eq!(got_a, ref_a, "a mismatch (n_src={})", n_src);
                assert_eq!(got_b, ref_b, "b mismatch (n_src={})", n_src);
            }
        }
    }

    /// Microbench: fused reduce+next-eval vs (reduce_only + standalone
    /// evaluate). Run with:
    ///
    ///   cargo test --release --lib bench_ext3_fused -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_ext3_fused_vs_separate() {
        use crate::tests::F64Ext3;
        use ark_ff::UniformRand;
        use std::time::Instant;

        let mut rng = test_rng();
        for num_vars in [16usize, 18, 20, 22, 24] {
            let n_src = 1usize << num_vars;
            let f: Vec<F64Ext3> = (0..n_src).map(|_| F64Ext3::rand(&mut rng)).collect();
            let g: Vec<F64Ext3> = (0..n_src).map(|_| F64Ext3::rand(&mut rng)).collect();
            let (f_c0, f_c1, f_c2): (Vec<u64>, Vec<u64>, Vec<u64>) = {
                let mut c0 = Vec::with_capacity(n_src);
                let mut c1 = Vec::with_capacity(n_src);
                let mut c2 = Vec::with_capacity(n_src);
                for x in &f {
                    let bytes: [u64; 3] = unsafe { *(x as *const F64Ext3 as *const [u64; 3]) };
                    c0.push(bytes[0]);
                    c1.push(bytes[1]);
                    c2.push(bytes[2]);
                }
                (c0, c1, c2)
            };
            let (g_c0, g_c1, g_c2): (Vec<u64>, Vec<u64>, Vec<u64>) = {
                let mut c0 = Vec::with_capacity(n_src);
                let mut c1 = Vec::with_capacity(n_src);
                let mut c2 = Vec::with_capacity(n_src);
                for x in &g {
                    let bytes: [u64; 3] = unsafe { *(x as *const F64Ext3 as *const [u64; 3]) };
                    c0.push(bytes[0]);
                    c1.push(bytes[1]);
                    c2.push(bytes[2]);
                }
                (c0, c1, c2)
            };

            let chg: [u64; 3] = {
                let c = F64Ext3::rand(&mut rng);
                unsafe { *(&c as *const F64Ext3 as *const [u64; 3]) }
            };
            let w: u64 = {
                let nr = F64Ext3::rand(&mut rng);
                unsafe { *(&nr as *const F64Ext3 as *const u64) }
            };

            // SEPARATE: reduce_only + standalone evaluate
            let mut out_f = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            let mut out_g = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            let t0 = Instant::now();
            ext3_soa_product_reduce_only_into::<Backend>(
                &f_c0,
                &f_c1,
                &f_c2,
                &g_c0,
                &g_c1,
                &g_c2,
                &mut out_f.0,
                &mut out_f.1,
                &mut out_f.2,
                &mut out_g.0,
                &mut out_g.1,
                &mut out_g.2,
                chg,
                w,
            );
            let _ = ext3_soa_product_evaluate::<Backend>(
                &out_f.0, &out_f.1, &out_f.2, &out_g.0, &out_g.1, &out_g.2, w,
            );
            let t_sep = t0.elapsed();

            // FUSED: reduce + next-eval in one pass
            let mut out_f = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            let mut out_g = (
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
                vec![0u64; n_src / 2],
            );
            let t0 = Instant::now();
            let _ = ext3_soa_product_fused_reduce_next_eval_into::<Backend>(
                &f_c0,
                &f_c1,
                &f_c2,
                &g_c0,
                &g_c1,
                &g_c2,
                &mut out_f.0,
                &mut out_f.1,
                &mut out_f.2,
                &mut out_g.0,
                &mut out_g.1,
                &mut out_g.2,
                chg,
                w,
            );
            let t_fused = t0.elapsed();

            let ratio = t_sep.as_secs_f64() / t_fused.as_secs_f64();
            println!(
                "num_vars={:>2}  n=2^{num_vars}  separate={:>10.3?}  fused={:>10.3?}  speedup={:.2}x",
                num_vars, t_sep, t_fused, ratio
            );
        }
    }
}
