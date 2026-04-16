//! Montgomery-form Goldilocks NEON backend.
//!
//! Operates directly on Montgomery-form values (as stored by arkworks `Fp64`),
//! enabling zero-cost `transmute` from `&[F64]` to `&[u64]`.
//!
//! Implements the same CIOS Montgomery reduction as arkworks' `MontBackend`
//! for `N=1`, so results are bit-identical.

use core::arch::aarch64::*;

use super::super::SimdBaseField;

/// Goldilocks modulus: P = 2^64 - 2^32 + 1.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Montgomery constant: INV = -P^{-1} mod 2^64.
const INV: u64 = 0xFFFF_FFFE_FFFF_FFFF;

/// ε = 2^64 mod P = 2^32 - 1 (used for add/sub overflow correction).
const EPSILON: u64 = 0xFFFF_FFFF;

/// Montgomery ONE = R mod P = 2^64 mod P = EPSILON.
const MONT_ONE: u64 = EPSILON;

/// Montgomery ZERO = 0 (same in both domains).
const MONT_ZERO: u64 = 0;

#[derive(Copy, Clone)]
pub struct GoldilocksNeon;

impl SimdBaseField for GoldilocksNeon {
    type Scalar = u64;
    type Packed = uint64x2_t;
    const LANES: usize = 2;
    const MODULUS: u64 = P;
    const ZERO: u64 = MONT_ZERO;
    const ONE: u64 = MONT_ONE;

    #[inline(always)]
    fn splat(val: u64) -> uint64x2_t {
        unsafe { vdupq_n_u64(val) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u64) -> uint64x2_t {
        unsafe { vld1q_u64(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut u64, v: uint64x2_t) {
        unsafe { vst1q_u64(ptr, v) }
    }

    #[inline(always)]
    unsafe fn load_deinterleaved(ptr: *const u64) -> (uint64x2_t, uint64x2_t) {
        let pair = unsafe { vld2q_u64(ptr) };
        (pair.0, pair.1)
    }

    // Add/sub are identical in canonical and Montgomery domain.
    // mont(a) + mont(b) = mont(a + b), same wrapping/reduction logic.

    #[inline(always)]
    fn add(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe {
            let sum = vaddq_u64(a, b);
            let p_vec = vdupq_n_u64(P);
            let eps_vec = vdupq_n_u64(EPSILON);
            let carry = vcltq_u64(sum, a);
            let geq_p = vcgeq_u64(sum, p_vec);
            let sub_p = vsubq_u64(sum, p_vec);
            let no_carry_result = vbslq_u64(geq_p, sub_p, sum);
            let carry_result = vaddq_u64(sum, eps_vec);
            vbslq_u64(carry, carry_result, no_carry_result)
        }
    }

    #[inline(always)]
    fn sub(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe {
            let diff = vsubq_u64(a, b);
            let p_vec = vdupq_n_u64(P);
            let borrow = vcltq_u64(a, b);
            let corrected = vaddq_u64(diff, p_vec);
            vbslq_u64(borrow, corrected, diff)
        }
    }

    #[inline(always)]
    fn mul(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // Per-lane Montgomery multiplication (CIOS for N=1).
        //
        // NEON has no 64×64→128 multiply instruction. We tried vectorizing
        // via four `vmull_u32` partial products (see `mont_mul_pair` below,
        // kept for testing/reference), but it was ~1.5× SLOWER across all
        // input sizes on Apple Silicon — the M-series scalar integer pipeline
        // is fast enough that `(a as u128) * (b as u128)` (compiled to
        // MUL+UMULH, 2 instructions) beats ~14+ NEON instructions for the
        // vectorized equivalent. On other ARM cores with narrower scalar
        // pipes (Graviton, Neoverse, older Cortex-A) the vectorized path
        // may still win; swap in `mont_mul_pair` there if benched as such.
        unsafe {
            let a0 = vgetq_lane_u64(a, 0);
            let a1 = vgetq_lane_u64(a, 1);
            let b0 = vgetq_lane_u64(b, 0);
            let b1 = vgetq_lane_u64(b, 1);

            let r0 = mont_mul(a0, b0);
            let r1 = mont_mul(a1, b1);

            vcombine_u64(vcreate_u64(r0), vcreate_u64(r1))
        }
    }

    #[inline(always)]
    fn scalar_add(a: u64, b: u64) -> u64 {
        let (sum, carry) = a.overflowing_add(b);
        if carry {
            sum + EPSILON
        } else if sum >= P {
            sum - P
        } else {
            sum
        }
    }

    #[inline(always)]
    fn scalar_sub(a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a.wrapping_sub(b).wrapping_add(P)
        }
    }

    #[inline(always)]
    fn scalar_mul(a: u64, b: u64) -> u64 {
        mont_mul(a, b)
    }
}

/// Montgomery multiplication for single-limb Goldilocks.
///
/// Computes `mont_mul(a, b) = a * b * R^{-1} mod P` where R = 2^64.
/// This is the CIOS algorithm for N=1, identical to arkworks' `MontBackend`.
///
///   full = a * b                           (128-bit)
///   lo = full mod 2^64
///   hi = full >> 64
///   k = lo * INV mod 2^64                  (INV = -P^{-1} mod 2^64)
///   t = k * P                              (128-bit)
///   result = (full + t) >> 64              (fits in 64 bits + carry)
///   if result >= P: result -= P
#[inline(always)]
fn mont_mul(a: u64, b: u64) -> u64 {
    let full = (a as u128) * (b as u128);
    let lo = full as u64;
    let hi = (full >> 64) as u64;

    // k = lo * INV mod 2^64
    let k = lo.wrapping_mul(INV);

    // t = k * P (128-bit)
    let t = (k as u128) * (P as u128);
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    // (full + t) >> 64 = hi + t_hi + carry_from(lo + t_lo)
    let (_, carry) = lo.overflowing_add(t_lo);
    let (mut result, carry2) = hi.overflowing_add(t_hi);
    let (result2, carry3) = result.overflowing_add(carry as u64);
    result = result2;

    // Handle carry: carry2 || carry3 can happen since Goldilocks has no spare bit
    if carry2 || carry3 || result >= P {
        result = result.wrapping_sub(P);
    }

    result
}

/// NEON-vectorized paired Montgomery multiply for two Goldilocks elements.
///
/// Input `a`, `b` each hold two u64 operands in Montgomery form. Returns
/// `[mont_mul(a[0], b[0]), mont_mul(a[1], b[1])]`.
///
/// 64×64→128 via four parallel `vmull_u32` instructions (each does 2 lanes
/// of 32×32→64), then CIOS Montgomery reduction using a second batch of
/// `vmull_u32`s for `k·P`. Two full Montgomery mults in ~20 NEON
/// instructions total.
///
/// **NOT currently wired into `F::mul`** on Apple Silicon: the scalar-
/// wrapped path (`(a as u128) * (b as u128)`, compiled to MUL+UMULH) is
/// faster on M-series because the scalar integer pipeline is very wide.
/// Kept here for reference + testing; plausibly wins on other ARM cores
/// (Graviton, Neoverse, Cortex-A78 and earlier) where scalar 64×64→128 is
/// more expensive. Bench before swapping in.
#[inline(always)]
#[allow(dead_code)]
unsafe fn mont_mul_pair(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    // ── Step 1: full 64×64→128 via schoolbook 32-bit partial products ──
    let a_lo32 = vmovn_u64(a);
    let a_hi32 = vshrn_n_u64::<32>(a);
    let b_lo32 = vmovn_u64(b);
    let b_hi32 = vshrn_n_u64::<32>(b);

    let ll = vmull_u32(a_lo32, b_lo32);
    let lh = vmull_u32(a_lo32, b_hi32);
    let hl = vmull_u32(a_hi32, b_lo32);
    let hh = vmull_u32(a_hi32, b_hi32);

    // Combine: full_128 = ll + (lh + hl) << 32 + hh << 64.
    // lh + hl may overflow u64; track the carry bit.
    let mid_lo = vaddq_u64(lh, hl);
    let mid_overflow = vcltq_u64(mid_lo, lh);
    let mid_carry = vshrq_n_u64::<63>(mid_overflow);

    // (mid << 32) split into (lo64, hi64):
    //   lo64 = mid_lo << 32 (mod 2^64)
    //   hi64 = (mid_lo >> 32) | (mid_carry << 32)
    let shifted_lo = vshlq_n_u64::<32>(mid_lo);
    let shifted_hi = vorrq_u64(vshrq_n_u64::<32>(mid_lo), vshlq_n_u64::<32>(mid_carry));

    // full_lo = ll + shifted_lo (with carry to full_hi)
    let full_lo = vaddq_u64(ll, shifted_lo);
    let full_lo_overflow = vcltq_u64(full_lo, ll);
    let full_lo_carry = vshrq_n_u64::<63>(full_lo_overflow);

    // full_hi = hh + shifted_hi + full_lo_carry. No further overflow
    // because (a*b) < 2^128 by construction, so full_hi < 2^64.
    let full_hi = vaddq_u64(vaddq_u64(hh, shifted_hi), full_lo_carry);

    // ── Step 2: k = (full_lo * INV) mod 2^64 ──
    // Only low 64 bits needed. 3 partial products suffice; the hh term
    // contributes to bits ≥ 64 and is dropped.
    let inv_vec = vdupq_n_u64(INV);
    let fl_lo32 = vmovn_u64(full_lo);
    let fl_hi32 = vshrn_n_u64::<32>(full_lo);
    let inv_lo32 = vmovn_u64(inv_vec);
    let inv_hi32 = vshrn_n_u64::<32>(inv_vec);

    let k_ll = vmull_u32(fl_lo32, inv_lo32);
    let k_lh = vmull_u32(fl_lo32, inv_hi32);
    let k_hl = vmull_u32(fl_hi32, inv_lo32);

    // k = k_ll + ((k_lh + k_hl) << 32) mod 2^64.
    let k_mid = vaddq_u64(k_lh, k_hl);
    let k = vaddq_u64(k_ll, vshlq_n_u64::<32>(k_mid));

    // ── Step 3: t = k * P (128-bit) via partial products ──
    // P = 2^64 − 2^32 + 1 → P.lo32 = 1, P.hi32 = 0xFFFFFFFF.
    let p_lo32 = vdup_n_u32(1u32);
    let p_hi32 = vdup_n_u32(0xFFFFFFFFu32);
    let k_lo32 = vmovn_u64(k);
    let k_hi32 = vshrn_n_u64::<32>(k);

    let t_ll = vmull_u32(k_lo32, p_lo32);
    let t_lh = vmull_u32(k_lo32, p_hi32);
    let t_hl = vmull_u32(k_hi32, p_lo32);
    let t_hh = vmull_u32(k_hi32, p_hi32);

    let t_mid_lo = vaddq_u64(t_lh, t_hl);
    let t_mid_overflow = vcltq_u64(t_mid_lo, t_lh);
    let t_mid_carry = vshrq_n_u64::<63>(t_mid_overflow);

    let t_shifted_lo = vshlq_n_u64::<32>(t_mid_lo);
    let t_shifted_hi = vorrq_u64(vshrq_n_u64::<32>(t_mid_lo), vshlq_n_u64::<32>(t_mid_carry));

    let t_lo = vaddq_u64(t_ll, t_shifted_lo);
    let t_lo_overflow = vcltq_u64(t_lo, t_ll);
    let t_lo_carry = vshrq_n_u64::<63>(t_lo_overflow);

    let t_hi = vaddq_u64(vaddq_u64(t_hh, t_shifted_hi), t_lo_carry);

    // ── Step 4: result = (full + t) >> 64 ──
    // By construction of k, (full_lo + t_lo) ≡ 0 (mod 2^64), so the
    // only information from the low 64 bits is the carry.
    let sum_lo = vaddq_u64(full_lo, t_lo);
    let sum_lo_overflow = vcltq_u64(sum_lo, full_lo);
    let sum_lo_carry = vshrq_n_u64::<63>(sum_lo_overflow);

    // result = full_hi + t_hi + sum_lo_carry. Can overflow u64 — track it.
    let result_tmp = vaddq_u64(full_hi, t_hi);
    let result_tmp_overflow = vcltq_u64(result_tmp, full_hi);
    let result = vaddq_u64(result_tmp, sum_lo_carry);
    let result_overflow = vcltq_u64(result, result_tmp);

    // Final overflow mask: either tmp overflowed, or the +carry overflowed.
    let total_overflow = vorrq_u64(result_tmp_overflow, result_overflow);

    // ── Step 5: final reduction, if overflowed or result ≥ P, subtract P ──
    let p_vec = vdupq_n_u64(P);
    let result_ge_p = vcgeq_u64(result, p_vec);
    let need_sub = vorrq_u64(total_overflow, result_ge_p);
    let result_sub = vsubq_u64(result, p_vec);

    vbslq_u64(need_sub, result_sub, result)
}

// ── Extension field SIMD multiply functions ─────────────────────────────────
//
// These are free functions rather than trait impls because the nonresidue
// is a runtime value (extracted from the arkworks extension field config
// during dispatch). The SimdExtField trait on mod.rs defines the interface;
// these functions implement the Karatsuba formulas for degree 2 and 3.

/// Degree-2 Karatsuba: (a0 + a1·X)(b0 + b1·X) mod (X² - w)
/// 3 base muls + 1 mul-by-w + adds.
#[inline(always)]
pub fn ext2_mul(a: [uint64x2_t; 2], b: [uint64x2_t; 2], w: uint64x2_t) -> [uint64x2_t; 2] {
    let v0 = GoldilocksNeon::mul(a[0], b[0]);
    let v1 = GoldilocksNeon::mul(a[1], b[1]);
    let c0 = GoldilocksNeon::add(v0, GoldilocksNeon::mul(w, v1));
    let a_sum = GoldilocksNeon::add(a[0], a[1]);
    let b_sum = GoldilocksNeon::add(b[0], b[1]);
    let c1 = GoldilocksNeon::sub(
        GoldilocksNeon::sub(GoldilocksNeon::mul(a_sum, b_sum), v0),
        v1,
    );
    [c0, c1]
}

/// Degree-2 Karatsuba (scalar version for tail processing).
#[inline(always)]
pub fn ext2_scalar_mul(a: [u64; 2], b: [u64; 2], w: u64) -> [u64; 2] {
    let v0 = mont_mul(a[0], b[0]);
    let v1 = mont_mul(a[1], b[1]);
    let c0 = GoldilocksNeon::scalar_add(v0, mont_mul(w, v1));
    let a_sum = GoldilocksNeon::scalar_add(a[0], a[1]);
    let b_sum = GoldilocksNeon::scalar_add(b[0], b[1]);
    let c1 = GoldilocksNeon::scalar_sub(GoldilocksNeon::scalar_sub(mont_mul(a_sum, b_sum), v0), v1);
    [c0, c1]
}

/// Degree-3 Karatsuba: (a0 + a1·X + a2·X²)(b0 + b1·X + b2·X²) mod (X³ - w)
/// 6 base muls + 2 mul-by-w + adds.
#[inline(always)]
pub fn ext3_mul(a: [uint64x2_t; 3], b: [uint64x2_t; 3], w: uint64x2_t) -> [uint64x2_t; 3] {
    let ad = GoldilocksNeon::mul(a[0], b[0]);
    let be = GoldilocksNeon::mul(a[1], b[1]);
    let cf = GoldilocksNeon::mul(a[2], b[2]);

    let x = GoldilocksNeon::sub(
        GoldilocksNeon::sub(
            GoldilocksNeon::mul(
                GoldilocksNeon::add(a[1], a[2]),
                GoldilocksNeon::add(b[1], b[2]),
            ),
            be,
        ),
        cf,
    );
    let y = GoldilocksNeon::sub(
        GoldilocksNeon::sub(
            GoldilocksNeon::mul(
                GoldilocksNeon::add(a[0], a[1]),
                GoldilocksNeon::add(b[0], b[1]),
            ),
            ad,
        ),
        be,
    );
    let z = GoldilocksNeon::add(
        GoldilocksNeon::sub(
            GoldilocksNeon::sub(
                GoldilocksNeon::mul(
                    GoldilocksNeon::add(a[0], a[2]),
                    GoldilocksNeon::add(b[0], b[2]),
                ),
                ad,
            ),
            cf,
        ),
        be,
    );

    [
        GoldilocksNeon::add(ad, GoldilocksNeon::mul(w, x)),
        GoldilocksNeon::add(y, GoldilocksNeon::mul(w, cf)),
        z,
    ]
}

/// Degree-3 Karatsuba (scalar version).
#[inline(always)]
pub fn ext3_scalar_mul(a: [u64; 3], b: [u64; 3], w: u64) -> [u64; 3] {
    let ad = mont_mul(a[0], b[0]);
    let be = mont_mul(a[1], b[1]);
    let cf = mont_mul(a[2], b[2]);

    let x = GoldilocksNeon::scalar_sub(
        GoldilocksNeon::scalar_sub(
            mont_mul(
                GoldilocksNeon::scalar_add(a[1], a[2]),
                GoldilocksNeon::scalar_add(b[1], b[2]),
            ),
            be,
        ),
        cf,
    );
    let y = GoldilocksNeon::scalar_sub(
        GoldilocksNeon::scalar_sub(
            mont_mul(
                GoldilocksNeon::scalar_add(a[0], a[1]),
                GoldilocksNeon::scalar_add(b[0], b[1]),
            ),
            ad,
        ),
        be,
    );
    let z = GoldilocksNeon::scalar_add(
        GoldilocksNeon::scalar_sub(
            GoldilocksNeon::scalar_sub(
                mont_mul(
                    GoldilocksNeon::scalar_add(a[0], a[2]),
                    GoldilocksNeon::scalar_add(b[0], b[2]),
                ),
                ad,
            ),
            cf,
        ),
        be,
    );

    [
        GoldilocksNeon::scalar_add(ad, mont_mul(w, x)),
        GoldilocksNeon::scalar_add(y, mont_mul(w, cf)),
        z,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{from_mont, to_mont, F64};
    use ark_ff::{AdditiveGroup, UniformRand};
    use ark_std::test_rng;

    #[test]
    fn test_mont_mul_matches_arkworks() {
        let mut rng = test_rng();
        for _ in 0..100_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);
            let expected = a * b;
            let result = from_mont(mont_mul(to_mont(a), to_mont(b)));
            assert_eq!(
                expected, result,
                "mont_mul mismatch for a={:?}, b={:?}",
                a, b
            );
        }
    }

    #[test]
    fn test_mont_add_matches_arkworks() {
        let mut rng = test_rng();
        for _ in 0..100_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);
            let expected = a + b;
            let result = from_mont(GoldilocksNeon::scalar_add(to_mont(a), to_mont(b)));
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn test_mont_sub_matches_arkworks() {
        let mut rng = test_rng();
        for _ in 0..100_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);
            let expected = a - b;
            let result = from_mont(GoldilocksNeon::scalar_sub(to_mont(a), to_mont(b)));
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn test_neon_mont_mul() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);

            let a_raw = [to_mont(a0), to_mont(a1)];
            let b_raw = [to_mont(b0), to_mont(b1)];

            let a_v = unsafe { GoldilocksNeon::load(a_raw.as_ptr()) };
            let b_v = unsafe { GoldilocksNeon::load(b_raw.as_ptr()) };
            let r_v = GoldilocksNeon::mul(a_v, b_v);

            let mut result = [0u64; 2];
            unsafe { GoldilocksNeon::store(result.as_mut_ptr(), r_v) };

            assert_eq!(from_mont(result[0]), a0 * b0);
            assert_eq!(from_mont(result[1]), a1 * b1);
        }
    }

    #[test]
    fn test_transmute_roundtrip() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let f = F64::rand(&mut rng);
            let mont = to_mont(f);
            let back = from_mont(mont);
            assert_eq!(f, back, "transmute roundtrip failed");
        }
    }

    #[test]
    fn test_edge_cases() {
        use ark_ff::Field;
        let zero = F64::ZERO;
        let one = F64::ONE;
        let neg_one = -F64::ONE;

        // 0 * anything = 0
        assert_eq!(from_mont(mont_mul(to_mont(zero), to_mont(neg_one))), zero);
        // 1 * x = x
        assert_eq!(from_mont(mont_mul(to_mont(one), to_mont(neg_one))), neg_one);
        // (-1) * (-1) = 1
        assert_eq!(from_mont(mont_mul(to_mont(neg_one), to_mont(neg_one))), one);
    }

    #[test]
    fn test_ext2_scalar_mul() {
        // Test degree-2 extension multiply against naive computation.
        // Using nonresidue w = 7 (in Montgomery form).
        let mut rng = test_rng();
        let w_mont = to_mont(F64::from(7u64));

        for _ in 0..10_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);

            let a = [to_mont(a0), to_mont(a1)];
            let b = [to_mont(b0), to_mont(b1)];
            let result = ext2_scalar_mul(a, b, w_mont);

            // Naive: c0 = a0*b0 + 7*a1*b1, c1 = a0*b1 + a1*b0
            let expected_c0 = a0 * b0 + F64::from(7u64) * a1 * b1;
            let expected_c1 = a0 * b1 + a1 * b0;

            assert_eq!(from_mont(result[0]), expected_c0, "ext2 c0 mismatch");
            assert_eq!(from_mont(result[1]), expected_c1, "ext2 c1 mismatch");
        }
    }

    #[test]
    fn test_ext3_scalar_mul() {
        // Test degree-3 extension multiply against naive schoolbook.
        // Using nonresidue w = 7 (in Montgomery form).
        let mut rng = test_rng();
        let w_mont = to_mont(F64::from(7u64));
        let w = F64::from(7u64);

        for _ in 0..10_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let a2 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);
            let b2 = F64::rand(&mut rng);

            let a = [to_mont(a0), to_mont(a1), to_mont(a2)];
            let b = [to_mont(b0), to_mont(b1), to_mont(b2)];
            let result = ext3_scalar_mul(a, b, w_mont);

            // Naive schoolbook mod (X³ - w):
            // c0 = a0*b0 + w*(a1*b2 + a2*b1)
            // c1 = a0*b1 + a1*b0 + w*a2*b2
            // c2 = a0*b2 + a1*b1 + a2*b0
            let expected_c0 = a0 * b0 + w * (a1 * b2 + a2 * b1);
            let expected_c1 = a0 * b1 + a1 * b0 + w * a2 * b2;
            let expected_c2 = a0 * b2 + a1 * b1 + a2 * b0;

            assert_eq!(from_mont(result[0]), expected_c0, "ext3 c0 mismatch");
            assert_eq!(from_mont(result[1]), expected_c1, "ext3 c1 mismatch");
            assert_eq!(from_mont(result[2]), expected_c2, "ext3 c2 mismatch");
        }
    }

    #[test]
    fn test_ext2_neon_matches_scalar() {
        // Verify NEON ext2_mul matches ext2_scalar_mul.
        let mut rng = test_rng();
        let w_mont = to_mont(F64::from(7u64));
        let w_vec = GoldilocksNeon::splat(w_mont);

        for _ in 0..10_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);

            let a_raw = [[to_mont(a0), to_mont(a0)], [to_mont(a1), to_mont(a1)]];
            let b_raw = [[to_mont(b0), to_mont(b0)], [to_mont(b1), to_mont(b1)]];

            let a_v = [unsafe { GoldilocksNeon::load(a_raw[0].as_ptr()) }, unsafe {
                GoldilocksNeon::load(a_raw[1].as_ptr())
            }];
            let b_v = [unsafe { GoldilocksNeon::load(b_raw[0].as_ptr()) }, unsafe {
                GoldilocksNeon::load(b_raw[1].as_ptr())
            }];

            let r_v = ext2_mul(a_v, b_v, w_vec);

            let mut r_out = [[0u64; 2]; 2];
            unsafe {
                GoldilocksNeon::store(r_out[0].as_mut_ptr(), r_v[0]);
                GoldilocksNeon::store(r_out[1].as_mut_ptr(), r_v[1]);
            }

            let scalar_result = ext2_scalar_mul(
                [to_mont(a0), to_mont(a1)],
                [to_mont(b0), to_mont(b1)],
                w_mont,
            );

            assert_eq!(r_out[0][0], scalar_result[0], "ext2 NEON c0 mismatch");
            assert_eq!(r_out[1][0], scalar_result[1], "ext2 NEON c1 mismatch");
        }
    }

    /// Fuzz `mont_mul_pair` against the scalar `mont_mul` reference.
    #[test]
    fn mont_mul_pair_matches_scalar() {
        use ark_std::{rand::Rng, test_rng};

        let mut rng = test_rng();

        // Deterministic corner cases first.
        let corners: [u64; 10] = [
            0,
            1,
            MONT_ONE,
            P - 1,
            P,
            0xFFFFFFFF_FFFFFFFF,
            0x8000_0000_0000_0000,
            0x7FFF_FFFF_FFFF_FFFF,
            EPSILON,
            INV,
        ];

        let mut check = |a0: u64, b0: u64, a1: u64, b1: u64| {
            // Operate on (mod P) reduced inputs — NEON backend expects
            // canonical Montgomery-form values in [0, P).
            let a0 = a0 % P;
            let b0 = b0 % P;
            let a1 = a1 % P;
            let b1 = b1 % P;

            let buf_a = [a0, a1];
            let buf_b = [b0, b1];
            let a_v = unsafe { vld1q_u64(buf_a.as_ptr()) };
            let b_v = unsafe { vld1q_u64(buf_b.as_ptr()) };

            let r_v = unsafe { mont_mul_pair(a_v, b_v) };
            let mut r_out = [0u64; 2];
            unsafe { vst1q_u64(r_out.as_mut_ptr(), r_v) };

            let ref0 = mont_mul(a0, b0);
            let ref1 = mont_mul(a1, b1);
            assert_eq!(
                r_out[0], ref0,
                "lane 0 mismatch: a={:016x} b={:016x} neon={:016x} ref={:016x}",
                a0, b0, r_out[0], ref0
            );
            assert_eq!(
                r_out[1], ref1,
                "lane 1 mismatch: a={:016x} b={:016x} neon={:016x} ref={:016x}",
                a1, b1, r_out[1], ref1
            );
        };

        // All corner × corner combinations (lane 0 only; lane 1 = random).
        for &a in corners.iter() {
            for &b in corners.iter() {
                let a1: u64 = rng.gen();
                let b1: u64 = rng.gen();
                check(a, b, a1, b1);
            }
        }

        // Fuzz 10k random pairs.
        for _ in 0..10_000 {
            let a0: u64 = rng.gen();
            let b0: u64 = rng.gen();
            let a1: u64 = rng.gen();
            let b1: u64 = rng.gen();
            check(a0, b0, a1, b1);
        }
    }
}
