//! Goldilocks NEON backend: packed `uint64x2_t` (2 lanes of u64).
//!
//! Goldilocks modulus: P = 2^64 - 2^32 + 1 = 0xFFFF_FFFF_0000_0001
//!
//! Key property: 2^64 ≡ 2^32 - 1 (mod P), so reduction of a 128-bit
//! product `(hi, lo)` is: `lo + hi * (2^32 - 1)`, with at most two
//! conditional subtractions.

use core::arch::aarch64::*;

use super::super::SimdBaseField;

/// Goldilocks field p = 2^64 - 2^32 + 1
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// ε = 2^32 - 1 = 0xFFFF_FFFF, used in reduction: 2^64 ≡ ε (mod P)... wait,
/// actually 2^64 = P + 2^32 - 1, so 2^64 ≡ 2^32 - 1 ≡ ε (mod P). Yes.
const EPSILON: u64 = 0xFFFF_FFFF;

#[derive(Copy, Clone)]
pub struct GoldilocksNeon;

impl SimdBaseField for GoldilocksNeon {
    type Scalar = u64;
    type Packed = uint64x2_t;
    const LANES: usize = 2;
    const MODULUS: u64 = P;
    const ZERO: u64 = 0;
    const ONE: u64 = 1;

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
    fn add(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // (a + b) mod P
        // Since a, b < P < 2^64, the sum can overflow u64.
        // Strategy: sum = a + b (wrapping). If sum < a, we overflowed.
        //   Overflowed: result = sum + EPSILON (since 2^64 ≡ ε mod P)
        //                        ... but we also need sum + ε < P check
        //   Not overflowed: if sum >= P then sum - P, else sum
        //
        // Equivalent (branchless): let (sum, carry) = a.overflowing_add(b);
        //   if carry: result = sum + EPSILON  (can't overflow again since a,b < P)
        //   else: if sum >= P then sum - P else sum
        //
        // NEON approach: use vaddq_u64 for wrapping add, then detect overflow
        // via vcltq_u64(sum, a) — if sum < a, overflow occurred.
        unsafe {
            let sum = vaddq_u64(a, b);
            let p_vec = vdupq_n_u64(P);
            let eps_vec = vdupq_n_u64(EPSILON);

            // Detect overflow: sum < a means carry occurred
            let carry = vcltq_u64(sum, a);
            // carry is all-ones (0xFFFF...) in lanes that overflowed

            // If carry: result = sum + EPSILON (overflow path)
            // If no carry and sum >= P: result = sum - P
            // If no carry and sum < P: result = sum

            // Non-overflow conditional subtract
            let geq_p = vcgeq_u64(sum, p_vec);
            let sub_p = vsubq_u64(sum, p_vec);

            // When no carry: choose between sum and sum-P
            let no_carry_result = vbslq_u64(geq_p, sub_p, sum);

            // When carry: sum + epsilon
            let carry_result = vaddq_u64(sum, eps_vec);

            // Select based on carry
            vbslq_u64(carry, carry_result, no_carry_result)
        }
    }

    #[inline(always)]
    fn sub(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // (a - b) mod P
        // If a >= b: result = a - b (may need no reduction since both < P)
        // If a < b: result = a - b + P (wrapping sub, then add P)
        //
        // But wrapping: diff = a.wrapping_sub(b). If a < b, diff "underflowed".
        // Detect: a < b via vcltq_u64(a, b).
        // Underflow path: diff + P. Since a,b < P, diff+P is in range.
        unsafe {
            let diff = vsubq_u64(a, b);
            let p_vec = vdupq_n_u64(P);

            // Detect underflow: a < b
            let borrow = vcltq_u64(a, b);

            // If borrow: diff + P. Otherwise: diff.
            let corrected = vaddq_u64(diff, p_vec);
            vbslq_u64(borrow, corrected, diff)
        }
    }

    #[inline(always)]
    fn mul(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // 64×64 → 128-bit multiply, then Goldilocks reduction.
        //
        // NEON doesn't have a 64×64→128 multiply instruction.
        // We decompose into 32-bit pieces:
        //   a = a_hi * 2^32 + a_lo
        //   b = b_hi * 2^32 + b_lo
        //   a*b = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32 + a_hi*b_hi*2^64
        //
        // Since 2^64 ≡ ε (mod P) and 2^32 ≡ 2^32 (mod P, since 2^32 < P):
        //   a*b ≡ a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32 + a_hi*b_hi*ε (mod P)
        //
        // But we need to be careful with carries. It's simpler and more robust
        // to compute the full 128-bit product and then reduce.
        //
        // We process each lane separately since NEON can't do 64×64→128 in one go.
        unsafe {
            // Extract lanes, multiply, reduce, repack
            let a0 = vgetq_lane_u64(a, 0);
            let a1 = vgetq_lane_u64(a, 1);
            let b0 = vgetq_lane_u64(b, 0);
            let b1 = vgetq_lane_u64(b, 1);

            let r0 = goldilocks_mul_scalar(a0, b0);
            let r1 = goldilocks_mul_scalar(a1, b1);

            vcombine_u64(vcreate_u64(r0), vcreate_u64(r1))
        }
    }

    #[inline(always)]
    fn scalar_add(a: u64, b: u64) -> u64 {
        let (sum, carry) = a.overflowing_add(b);
        if carry {
            // 2^64 ≡ ε (mod P)
            sum + EPSILON // can't overflow again since a, b < P
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
        goldilocks_mul_scalar(a, b)
    }
}

/// Full 64×64 → 128-bit multiply with Goldilocks reduction.
///
/// Computes `(a * b) mod P` where P = 2^64 - 2^32 + 1.
///
/// Uses the identity: if a*b = hi * 2^64 + lo, then
///   a*b ≡ lo + hi * ε (mod P)  where ε = 2^32 - 1
///
/// Since hi < 2^64 and ε < 2^32, the product hi * ε < 2^96,
/// so we need to handle the intermediate result carefully.
#[inline(always)]
fn goldilocks_mul_scalar(a: u64, b: u64) -> u64 {
    let full = (a as u128) * (b as u128);
    let lo = full as u64;
    let hi = (full >> 64) as u64;
    goldilocks_reduce(lo, hi)
}

/// Reduce a 128-bit value `(lo + hi * 2^64)` modulo P = 2^64 - 2^32 + 1.
///
/// Using 2^64 ≡ ε (mod P) where ε = 2^32 - 1:
///   result ≡ lo + hi * ε (mod P)
///
/// We compute hi * ε = hi * (2^32 - 1) = (hi << 32) - hi,
/// carefully handling the intermediate 96-bit value.
#[inline(always)]
fn goldilocks_reduce(lo: u64, hi: u64) -> u64 {
    // hi * ε = hi * 2^32 - hi
    // Split: hi_hi = hi >> 32, hi_lo = hi & 0xFFFF_FFFF
    // hi * 2^32 = hi_lo * 2^32 + hi_hi * 2^64
    //           ≡ hi_lo * 2^32 + hi_hi * ε  (mod P)
    //  ... this recurses. Better: direct computation.
    //
    // Let's compute step by step:
    //   hi * ε where ε = 2^32 - 1
    //   = (hi << 32) - hi
    //
    // (hi << 32) can produce a 96-bit value. Let:
    //   hi_hi = hi >> 32
    //   hi_lo = hi & 0xFFFF_FFFF
    //
    //   hi << 32 = hi_lo << 32 | 0  (low 64 bits) + hi_hi (carry into 2^64)
    //
    // So: hi * 2^32 = (hi_lo << 32) + hi_hi * 2^64
    //     hi * ε = (hi_lo << 32) + hi_hi * 2^64 - hi
    //            ≡ (hi_lo << 32) + hi_hi * ε - hi    (mod P)
    //
    // Since hi_hi < 2^32 and ε < 2^32, hi_hi * ε < 2^64, fits in u64.
    //
    // Total: lo + (hi_lo << 32) + hi_hi * ε - hi  (mod P)
    //
    // This can still overflow, so we need careful addition.

    let hi_hi = hi >> 32;
    let hi_lo = hi & 0xFFFF_FFFF;

    // term1 = hi_lo << 32 (fits in u64, since hi_lo < 2^32)
    let term1 = hi_lo << 32;

    // term2 = hi_hi * EPSILON (fits in u64, since hi_hi < 2^32, EPSILON < 2^32)
    let term2 = hi_hi * EPSILON;

    // result = lo + term1 + term2 - hi (mod P)
    // Do additions first, then subtraction, with overflow handling.

    // lo + term1
    let (s1, c1) = lo.overflowing_add(term1);
    // s1 + term2
    let (s2, c2) = s1.overflowing_add(term2);
    // Total carry count (0, 1, or 2). Each carry means +ε in the final result.
    let carry = (c1 as u64) + (c2 as u64);

    // s2 + carry * EPSILON - hi
    // First: s2 + carry * EPSILON
    let (s3, c3) = s2.overflowing_add(carry * EPSILON);
    let carry2 = c3 as u64;

    // Now subtract hi
    let (s4, borrow) = s3.overflowing_sub(hi);
    let borrow_val = borrow as u64;

    // Net adjustment: carry2 * EPSILON - borrow_val * P
    // But since carry2 ∈ {0,1} and borrow ∈ {0,1}, let's handle:
    // result = s4 + carry2 * EPSILON (from overflow in s3)
    //            + borrow_val * P    (to compensate underflow in s4)
    // Wait: if borrow, the true value is s4 + 2^64 - hi_val = s4 + ε (mod P).
    // No: s3 - hi. If borrow, true value = s3 - hi + 2^64 ≡ s4 + ε + 1 (mod P)?
    // 2^64 mod P = ε + 1? No. 2^64 = P + 2^32 - 1 = P + ε, so 2^64 ≡ ε (mod P).
    // Hmm, P = 2^64 - 2^32 + 1, so 2^64 = P + 2^32 - 1 = P + ε.
    // So 2^64 ≡ ε ≡ EPSILON (mod P). ← Wait that's wrong.
    // P = 2^64 - ε - 1? Let me recheck. P = 2^64 - 2^32 + 1.
    // 2^64 = P + 2^32 - 1 = P + EPSILON.
    // So 2^64 mod P = EPSILON.
    // No wait: EPSILON = 2^32 - 1 = 0xFFFF_FFFF.
    // P = 2^64 - 2^32 + 1 = 2^64 - EPSILON - 1.
    // Hmm, 2^64 = P + EPSILON + 1? Let me just compute:
    // 2^64 - P = 2^64 - (2^64 - 2^32 + 1) = 2^32 - 1 = EPSILON.
    // So 2^64 ≡ EPSILON (mod P)? No:
    // 2^64 = 1 * P + EPSILON. So 2^64 mod P = EPSILON. ← Wait:
    // P = 18446744069414584321
    // 2^64 = 18446744073709551616
    // 2^64 - P = 18446744073709551616 - 18446744069414584321 = 4294967295 = 0xFFFF_FFFF = EPSILON
    // Yes! 2^64 mod P = EPSILON.

    // So if s3 overflowed (carry2=1), add EPSILON.
    // If subtraction borrowed (borrow_val=1), we need s4 + 2^64 (mod P) => s4 + EPSILON.
    // Wait, that's not right either. Borrow means the true mathematical result is
    // s3 - hi + 2^64. mod P, that's s4 + EPSILON.
    //
    // But the carry2 case: s2 + carry*ε overflowed, so the true value is
    // s3 + 2^64. mod P, that's s3 + EPSILON.
    // We already set s3 = the overflow result, and carry2 flags the overflow.
    //
    // Net: s4 + carry2 * EPSILON + borrow_val * EPSILON
    //    = s4 + (carry2 + borrow_val) * EPSILON
    //
    // Hmm, but borrow means we subtracted too much, so we should ADD back, not add EPSILON.
    // Let me re-derive:
    //   After carry step: true value = s3 + carry2 * 2^64 ≡ s3 + carry2 * EPSILON (mod P)
    //   After sub step:   true value = (s3 + carry2*2^64) - hi
    //     = s4 + borrow * 2^64 + carry2 * 2^64
    //     Wait no. Let me be more careful.
    //
    //   Let V = s2 + carry * EPSILON  (mathematical, could be > 2^64)
    //   s3 = V mod 2^64, carry2 = V >= 2^64
    //   So V = s3 + carry2 * 2^64
    //
    //   Let W = V - hi = s3 + carry2 * 2^64 - hi
    //   s4 = s3.wrapping_sub(hi), borrow = s3 < hi
    //   s4 = s3 - hi + borrow * 2^64
    //   So s3 - hi = s4 - borrow * 2^64
    //   W = s4 - borrow * 2^64 + carry2 * 2^64
    //     = s4 + (carry2 - borrow) * 2^64
    //     ≡ s4 + (carry2 - borrow) * EPSILON (mod P)
    //
    // carry2 - borrow ∈ {-1, 0, 1}
    // If +1: add EPSILON
    // If 0: done
    // If -1: subtract EPSILON (equivalently, add P - EPSILON = 2^64 - 2*EPSILON)
    //   ... but simpler: add P (since subtracting EPSILON when result could underflow)

    let adj = (carry2 as i64) - (borrow_val as i64);
    if adj > 0 {
        let (r, overflow) = s4.overflowing_add(EPSILON);
        if overflow || r >= P {
            r.wrapping_sub(P)
        } else {
            r
        }
    } else if adj < 0 {
        // s4 - EPSILON; if underflow, add P
        if s4 >= EPSILON {
            let r = s4 - EPSILON;
            if r >= P {
                r - P
            } else {
                r
            }
        } else {
            // s4 - EPSILON + P = s4 + (P - EPSILON) = s4 + 2^64 - 2*EPSILON
            // P - EPSILON = 2^64 - 2^32 + 1 - (2^32 - 1) = 2^64 - 2^33 + 2
            // That doesn't look right for a simple formula. Let's just do:
            s4.wrapping_sub(EPSILON).wrapping_add(P)
        }
    } else {
        // adj == 0
        if s4 >= P {
            s4 - P
        } else {
            s4
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    // Use the existing Goldilocks (F64) field from the test module for reference
    use crate::tests::F64;

    /// Convert an arkworks F64 element to its raw u64 representative in [0, P).
    fn to_raw(f: F64) -> u64 {
        use ark_ff::PrimeField;
        // BigInt -> u64
        let big = f.into_bigint();
        big.0[0]
    }

    #[test]
    fn test_scalar_add() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);
            let expected = to_raw(a + b);
            let received = GoldilocksNeon::scalar_add(to_raw(a), to_raw(b));
            assert_eq!(
                expected,
                received,
                "add failed for a={}, b={}",
                to_raw(a),
                to_raw(b)
            );
        }
    }

    #[test]
    fn test_scalar_sub() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);
            let expected = to_raw(a - b);
            let received = GoldilocksNeon::scalar_sub(to_raw(a), to_raw(b));
            assert_eq!(
                expected,
                received,
                "sub failed for a={}, b={}",
                to_raw(a),
                to_raw(b)
            );
        }
    }

    #[test]
    fn test_scalar_mul() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);
            let expected = to_raw(a * b);
            let received = GoldilocksNeon::scalar_mul(to_raw(a), to_raw(b));
            assert_eq!(
                expected,
                received,
                "mul failed for a={}, b={}",
                to_raw(a),
                to_raw(b)
            );
        }
    }

    #[test]
    fn test_neon_add() {
        let mut rng = test_rng();
        for _ in 0..5_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);

            let a_raw = [to_raw(a0), to_raw(a1)];
            let b_raw = [to_raw(b0), to_raw(b1)];

            let a_v = unsafe { GoldilocksNeon::load(a_raw.as_ptr()) };
            let b_v = unsafe { GoldilocksNeon::load(b_raw.as_ptr()) };
            let r_v = GoldilocksNeon::add(a_v, b_v);

            let mut result = [0u64; 2];
            unsafe { GoldilocksNeon::store(result.as_mut_ptr(), r_v) };

            assert_eq!(result[0], to_raw(a0 + b0));
            assert_eq!(result[1], to_raw(a1 + b1));
        }
    }

    #[test]
    fn test_neon_sub() {
        let mut rng = test_rng();
        for _ in 0..5_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);

            let a_raw = [to_raw(a0), to_raw(a1)];
            let b_raw = [to_raw(b0), to_raw(b1)];

            let a_v = unsafe { GoldilocksNeon::load(a_raw.as_ptr()) };
            let b_v = unsafe { GoldilocksNeon::load(b_raw.as_ptr()) };
            let r_v = GoldilocksNeon::sub(a_v, b_v);

            let mut result = [0u64; 2];
            unsafe { GoldilocksNeon::store(result.as_mut_ptr(), r_v) };

            assert_eq!(result[0], to_raw(a0 - b0));
            assert_eq!(result[1], to_raw(a1 - b1));
        }
    }

    #[test]
    fn test_neon_mul() {
        let mut rng = test_rng();
        for _ in 0..5_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);

            let a_raw = [to_raw(a0), to_raw(a1)];
            let b_raw = [to_raw(b0), to_raw(b1)];

            let a_v = unsafe { GoldilocksNeon::load(a_raw.as_ptr()) };
            let b_v = unsafe { GoldilocksNeon::load(b_raw.as_ptr()) };
            let r_v = GoldilocksNeon::mul(a_v, b_v);

            let mut result = [0u64; 2];
            unsafe { GoldilocksNeon::store(result.as_mut_ptr(), r_v) };

            assert_eq!(result[0], to_raw(a0 * b0));
            assert_eq!(result[1], to_raw(a1 * b1));
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test with boundary values
        let zero = 0u64;
        let one = 1u64;
        let p_minus_1 = P - 1;

        // 0 + 0 = 0
        assert_eq!(GoldilocksNeon::scalar_add(zero, zero), zero);
        // 0 * anything = 0
        assert_eq!(GoldilocksNeon::scalar_mul(zero, p_minus_1), zero);
        // 1 * x = x
        assert_eq!(GoldilocksNeon::scalar_mul(one, p_minus_1), p_minus_1);
        // (P-1) + 1 = 0
        assert_eq!(GoldilocksNeon::scalar_add(p_minus_1, one), zero);
        // 0 - 1 = P - 1
        assert_eq!(GoldilocksNeon::scalar_sub(zero, one), p_minus_1);
        // (P-1) * (P-1) = 1
        assert_eq!(GoldilocksNeon::scalar_mul(p_minus_1, p_minus_1), one);
    }
}
