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
