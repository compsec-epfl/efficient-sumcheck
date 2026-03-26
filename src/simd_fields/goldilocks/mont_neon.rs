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
pub struct MontGoldilocksNeon;

impl SimdBaseField for MontGoldilocksNeon {
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
        // NEON has no 64×64→128, so we extract lanes and use scalar.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::F64;
    use ark_ff::{AdditiveGroup, BigInt, Field, UniformRand};
    use ark_std::test_rng;
    use core::marker::PhantomData;

    /// Get the Montgomery-form value (raw internal representation).
    fn to_mont(f: F64) -> u64 {
        (f.0).0[0]
    }

    /// Reconstruct F64 from Montgomery-form value.
    fn from_mont(val: u64) -> F64 {
        ark_ff::Fp(BigInt([val]), PhantomData)
    }

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
            let result = from_mont(MontGoldilocksNeon::scalar_add(to_mont(a), to_mont(b)));
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
            let result = from_mont(MontGoldilocksNeon::scalar_sub(to_mont(a), to_mont(b)));
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

            let a_v = unsafe { MontGoldilocksNeon::load(a_raw.as_ptr()) };
            let b_v = unsafe { MontGoldilocksNeon::load(b_raw.as_ptr()) };
            let r_v = MontGoldilocksNeon::mul(a_v, b_v);

            let mut result = [0u64; 2];
            unsafe { MontGoldilocksNeon::store(result.as_mut_ptr(), r_v) };

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
}
