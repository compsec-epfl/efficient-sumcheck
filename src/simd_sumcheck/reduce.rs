#![allow(dead_code)]
//! SIMD-vectorized reduce kernels: fold evaluations with a challenge.
//!
//! Two layout variants:
//!   - **Half-split (MSB)**: pairs `data[k]` with `data[k + L/2]`. This is
//!     the layout used by the public sumcheck entry points and by WHIR.
//!   - **Pair-split (LSB)**: pairs `data[2k]` with `data[2k+1]`. Used by the
//!     legacy `Prover` trait and `coefficient_sumcheck`.
//!
//! The MSB kernel (`reduce_msb_in_place`) uses plain contiguous `F::load`
//! from each half — simpler and faster than the LSB `load_deinterleaved`.

use crate::simd_fields::SimdBaseField;

// ═══════════════════════════════════════════════════════════════════════════
// Half-split (MSB) reduce
// ═══════════════════════════════════════════════════════════════════════════

/// SIMD-vectorized MSB (half-split) reduce, in-place.
///
/// `new[k] = src[k] + challenge * (src[k + half] − src[k])` for `k` in
/// `0..half`, where `half = next_power_of_two(n) / 2`. Elements in the low
/// half beyond `n − half` (the "tail") have no partner in the high half and
/// are folded as `src[k] * (1 − challenge)`.
///
/// Returns the output length `half`.
pub fn reduce_msb_in_place<F: SimdBaseField>(src: &mut [F::Scalar], challenge: F::Scalar) -> usize {
    let n = src.len();
    if n <= 1 {
        return n;
    }

    let half = n.next_power_of_two() >> 1;
    let paired = n - half; // elements that have a partner in the high half
    let lanes = F::LANES;
    let challenge_v = F::splat(challenge);

    // ── SIMD main loop over paired portion ──
    let step = 4 * lanes;
    let aligned = (paired / step) * step;

    let lo_ptr = src.as_ptr();
    let hi_ptr = unsafe { src.as_ptr().add(half) };
    let out_ptr = src.as_mut_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            for g in 0..4 {
                let off = i + g * lanes;
                let a = F::load(lo_ptr.add(off));
                let b = F::load(hi_ptr.add(off));
                let r = F::add(a, F::mul(challenge_v, F::sub(b, a)));
                F::store(out_ptr.add(off), r);
            }
        }
        i += step;
    }

    // ── Scalar tail of paired portion ──
    while i < paired {
        let a = src[i];
        let b = src[i + half];
        src[i] = F::scalar_add(a, F::scalar_mul(challenge, F::scalar_sub(b, a)));
        i += 1;
    }

    // ── Unpaired tail: data[k] *= (1 − challenge) for k in paired..half ──
    let one_minus = F::scalar_sub(F::ONE, challenge);
    for v in src.iter_mut().take(half).skip(paired) {
        *v = F::scalar_mul(*v, one_minus);
    }

    half
}

// ═══════════════════════════════════════════════════════════════════════════
// Pair-split (LSB) reduce — legacy, used by coefficient_sumcheck
// ═══════════════════════════════════════════════════════════════════════════

/// SIMD-vectorized pairwise reduce, producing a new Vec.
///
/// Uses 4× loop unrolling for instruction-level parallelism.
/// (8× was benchmarked but regressed due to register pressure from mul.)
/// Stack-allocated deinterleave buffers avoid per-iteration heap allocation.
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
            F::store(out_ptr.add(i), r);
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
    debug_assert!(F::LANES <= 32);
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
#[allow(dead_code)]
#[cfg_attr(
    not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )),
    allow(unused_variables)
)]
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
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
#[cfg_attr(
    not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )),
    allow(unused_variables)
)]
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
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
    use crate::tests::F64;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_reduce_and_evaluate_matches() {
        use crate::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 16;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut evals_raw: Vec<u64> = evals_ff.iter().map(|f| (*f).value).collect();

        let challenge_ff = F64::rand(&mut rng);
        let challenge_raw = challenge_ff.value;

        // Reference: reduce then evaluate
        let mut expected_ff = evals_ff;
        pairwise::reduce_evaluations(&mut expected_ff, challenge_ff);
        let (expected_even, expected_odd) = pairwise::evaluate(&expected_ff);

        // Fused
        let (fused_even, fused_odd, new_len) =
            reduce_and_evaluate::<Backend>(&mut evals_raw, challenge_raw);

        assert_eq!(new_len, n / 2);
        assert_eq!(expected_even.value, fused_even, "fused even mismatch");
        assert_eq!(expected_odd.value, fused_odd, "fused odd mismatch");

        // Also verify the reduce output matches
        for i in 0..new_len {
            assert_eq!(
                expected_ff[i].value, evals_raw[i],
                "reduce mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_reduce_and_evaluate_large() {
        use crate::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 20;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut evals_raw: Vec<u64> = evals_ff.iter().map(|f| (*f).value).collect();

        let challenge_ff = F64::rand(&mut rng);
        let challenge_raw = challenge_ff.value;

        let mut expected_ff = evals_ff;
        pairwise::reduce_evaluations(&mut expected_ff, challenge_ff);
        let (expected_even, expected_odd) = pairwise::evaluate(&expected_ff);

        let (fused_even, fused_odd, _) =
            reduce_and_evaluate::<Backend>(&mut evals_raw, challenge_raw);

        assert_eq!(expected_even.value, fused_even, "large fused even mismatch");
        assert_eq!(expected_odd.value, fused_odd, "large fused odd mismatch");
    }
}
