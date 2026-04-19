#![allow(dead_code)]
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
}
