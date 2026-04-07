//! Montgomery-form Goldilocks AVX-512 IFMA backend.
//!
//! Operates directly on Montgomery-form values (as stored by arkworks `Fp64`),
//! enabling zero-cost `transmute` from `&[F64]` to `&[u64]`.
//!
//! Uses AVX-512 IFMA (52-bit multiply-accumulate) for true 8-wide vectorized
//! Montgomery multiplication. Unlike the NEON backend (which falls back to
//! scalar mont_mul per lane because NEON lacks 64x64->128 multiply), this
//! backend decomposes operands into 52-bit limbs and uses `vpmadd52luq` /
//! `vpmadd52huq` for a fully vectorized schoolbook multiply. Montgomery
//! reduction exploits the Goldilocks prime structure (P = 2^64 - 2^32 + 1)
//! to avoid additional IFMA multiplies — only shifts, adds, and subtracts.

use core::arch::x86_64::*;

use super::super::SimdBaseField;

/// Goldilocks modulus: P = 2^64 - 2^32 + 1.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// ε = 2^64 mod P = 2^32 - 1 (used for add/sub overflow correction).
const EPSILON: u64 = 0xFFFF_FFFF;

/// Montgomery constant: INV = -P^{-1} mod 2^64.
const INV: u64 = 0xFFFF_FFFE_FFFF_FFFF;

/// Montgomery ONE = R mod P = 2^64 mod P = EPSILON.
const MONT_ONE: u64 = EPSILON;

/// Montgomery ZERO = 0 (same in both domains).
const MONT_ZERO: u64 = 0;

/// Mask for lower 52 bits (IFMA operand width).
const MASK52: u64 = (1u64 << 52) - 1;

#[derive(Copy, Clone)]
pub struct GoldilocksAvx512;

impl SimdBaseField for GoldilocksAvx512 {
    type Scalar = u64;
    type Packed = __m512i;
    const LANES: usize = 8;
    const MODULUS: u64 = P;
    const ZERO: u64 = MONT_ZERO;
    const ONE: u64 = MONT_ONE;

    #[inline(always)]
    fn splat(val: u64) -> __m512i {
        unsafe { _mm512_set1_epi64(val as i64) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u64) -> __m512i {
        unsafe { _mm512_loadu_si512(ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut u64, v: __m512i) {
        unsafe { _mm512_storeu_si512(ptr.cast(), v) }
    }

    // Add/sub are identical in canonical and Montgomery domain.

    #[inline(always)]
    fn add(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let sum = _mm512_add_epi64(a, b);
            let p_vec = _mm512_set1_epi64(P as i64);
            let eps_vec = _mm512_set1_epi64(EPSILON as i64);

            // Detect unsigned overflow: sum < a means carry occurred
            let carry = _mm512_cmplt_epu64_mask(sum, a);
            // Detect sum >= P (only relevant when no carry)
            let ge_p = !_mm512_cmplt_epu64_mask(sum, p_vec); // >= is NOT <

            // Carry path: sum + ε (2^64 ≡ ε mod P, result guaranteed < P)
            let result = _mm512_mask_add_epi64(sum, carry, sum, eps_vec);
            // No-carry, >= P path: sum - P
            let need_sub = ge_p & !carry;
            _mm512_mask_sub_epi64(result, need_sub, result, p_vec)
        }
    }

    #[inline(always)]
    fn sub(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            let diff = _mm512_sub_epi64(a, b);
            let p_vec = _mm512_set1_epi64(P as i64);
            // Borrow when a < b (unsigned)
            let borrow = _mm512_cmplt_epu64_mask(a, b);
            _mm512_mask_add_epi64(diff, borrow, diff, p_vec)
        }
    }

    #[inline(always)]
    fn mul(a: __m512i, b: __m512i) -> __m512i {
        // True 8-wide Montgomery multiplication via IFMA 52-bit decomposition.
        //
        // 1. Schoolbook 64×64→128 product using 52-bit limbs + IFMA
        // 2. Montgomery reduction factor m via Goldilocks structure:
        //    INV = -(2^32+1) mod 2^64, so m = -(lo + lo<<32) — no multiply
        // 3. m*P via P = 2^64 - 2^32 + 1 — shifts and subtracts only
        // 4. result = (product + m*P) >> 64, conditional subtract P
        unsafe { avx512_mont_mul(a, b) }
    }

    #[inline(always)]
    fn add_wrapping(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi64(a, b) }
    }

    #[inline(always)]
    fn carry_mask(sum: __m512i, a_before: __m512i) -> __m512i {
        unsafe {
            let carry = _mm512_cmplt_epu64_mask(sum, a_before);
            _mm512_maskz_set1_epi64(carry, 1)
        }
    }

    #[inline(always)]
    fn reduce_carry(sum: __m512i, carry_count: __m512i) -> __m512i {
        // Each carry represents 2^64 ≡ EPSILON (mod P).
        // correction = carry_count * EPSILON (fits in u64 for reasonable counts).
        unsafe {
            let eps_vec = _mm512_set1_epi64(EPSILON as i64);
            let correction = _mm512_mullo_epi64(carry_count, eps_vec);
            Self::add(sum, correction)
        }
    }

    #[inline(always)]
    unsafe fn load_deinterleaved(ptr: *const u64) -> (__m512i, __m512i) {
        unsafe {
            let v0 = _mm512_loadu_si512(ptr.cast()); // [a0,b0,a1,b1,a2,b2,a3,b3]
            let v1 = _mm512_loadu_si512(ptr.add(8).cast()); // [a4,b4,a5,b5,a6,b6,a7,b7]
            let idx_even = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
            let idx_odd = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
            let evens = _mm512_permutex2var_epi64(v0, idx_even, v1);
            let odds = _mm512_permutex2var_epi64(v0, idx_odd, v1);
            (evens, odds)
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

/// AVX-512 IFMA Montgomery multiplication (8-wide).
///
/// Decomposes each 64-bit operand into two 52-bit limbs, performs a
/// schoolbook multiply using `vpmadd52luq`/`vpmadd52huq` (6 IFMA ops),
/// then reduces via the Goldilocks prime structure using only shifts,
/// adds, and masked operations — no additional multiplies needed.
#[inline(always)]
unsafe fn avx512_mont_mul(a: __m512i, b: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let mask52_vec = _mm512_set1_epi64(MASK52 as i64);
    let p_vec = _mm512_set1_epi64(P as i64);
    let ones = _mm512_set1_epi64(1);

    // ── Decompose into 52-bit limbs ──
    let a0 = _mm512_and_si512(a, mask52_vec); // low 52 bits
    let a1 = _mm512_srli_epi64(a, 52); // high 12 bits
    let b0 = _mm512_and_si512(b, mask52_vec);
    let b1 = _mm512_srli_epi64(b, 52);

    // ── Schoolbook multiply in base-2^52 (6 IFMA ops) ──
    // Limb 0: lo52(a0*b0) — exactly 52 bits
    let c0 = _mm512_madd52lo_epu64(zero, a0, b0);

    // Limb 1: hi52(a0*b0) + lo52(a0*b1) + lo52(a1*b0) — up to ~54 bits
    let c1 = _mm512_madd52hi_epu64(zero, a0, b0);
    let c1 = _mm512_madd52lo_epu64(c1, a0, b1);
    let c1 = _mm512_madd52lo_epu64(c1, a1, b0);

    // Limb 2: hi52(a0*b1) + hi52(a1*b0) + lo52(a1*b1) — up to ~25 bits
    let c2 = _mm512_madd52hi_epu64(zero, a0, b1);
    let c2 = _mm512_madd52hi_epu64(c2, a1, b0);
    let c2 = _mm512_madd52lo_epu64(c2, a1, b1);

    // ── Carry propagation: c1 → c2 ──
    let carry = _mm512_srli_epi64(c1, 52);
    let c1 = _mm512_and_si512(c1, mask52_vec); // now exactly 52 bits
    let c2 = _mm512_add_epi64(c2, carry);

    // ── Reconstruct (lo64, hi64) of the 128-bit product ──
    // lo64 = c0[0:51] | c1[0:11] << 52
    let lo = _mm512_or_si512(c0, _mm512_slli_epi64(c1, 52));
    // hi64 = c1[12:51] | c2 << 40 (non-overlapping since c1>>12 is 40 bits)
    let hi = _mm512_or_si512(_mm512_srli_epi64(c1, 12), _mm512_slli_epi64(c2, 40));

    // ── Montgomery reduction using Goldilocks structure ──
    //
    // m = lo * INV mod 2^64
    // INV = -(2^32 + 1) mod 2^64, so m = -(lo + lo<<32) — no multiply!
    let lo_shl32 = _mm512_slli_epi64(lo, 32);
    let temp = _mm512_add_epi64(lo, lo_shl32);
    let m = _mm512_sub_epi64(zero, temp);

    // m * P where P = 2^64 - 2^32 + 1:
    //   m*P = m*2^64 + m*(1 - 2^32)
    //   lo(m*P) = (m - m<<32) mod 2^64
    //   hi(m*P) = m - (m>>32) - borrow_from_lo
    //
    // The m*2^32 term spans two 64-bit words: hi = m>>32, lo = m<<32.
    let m_shl32 = _mm512_slli_epi64(m, 32);
    let m_shr32 = _mm512_srli_epi64(m, 32);
    let borrow_mask = _mm512_cmplt_epu64_mask(m, m_shl32);
    let hi_mp = _mm512_sub_epi64(m, m_shr32);
    let hi_mp = _mm512_mask_sub_epi64(hi_mp, borrow_mask, hi_mp, ones);

    // result = (product + m*P) >> 64
    // Since lo + lo(m*P) ≡ 0 mod 2^64 by construction, the carry is (lo != 0).
    let lo_nonzero = !_mm512_cmpeq_epu64_mask(lo, zero);
    let carry_from_lo = _mm512_maskz_set1_epi64(lo_nonzero, 1);

    // r = hi + hi(m*P) + carry
    let r1 = _mm512_add_epi64(hi, hi_mp);
    let c2_mask = _mm512_cmplt_epu64_mask(r1, hi); // overflow from first add

    let r2 = _mm512_add_epi64(r1, carry_from_lo);
    let c3_mask = _mm512_cmplt_epu64_mask(r2, r1); // overflow from second add

    // ── Final reduction: subtract P if carry or result >= P ──
    let ge_p = !_mm512_cmplt_epu64_mask(r2, p_vec);
    let need_sub = c2_mask | c3_mask | ge_p;
    _mm512_mask_sub_epi64(r2, need_sub, r2, p_vec)
}

/// Montgomery multiplication for single-limb Goldilocks (scalar).
///
/// Computes `mont_mul(a, b) = a * b * R^{-1} mod P` where R = 2^64.
/// CIOS algorithm for N=1, identical to arkworks' `MontBackend`.
#[inline(always)]
fn mont_mul(a: u64, b: u64) -> u64 {
    let full = (a as u128) * (b as u128);
    let lo = full as u64;
    let hi = (full >> 64) as u64;

    let k = lo.wrapping_mul(INV);

    let t = (k as u128) * (P as u128);
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    let (_, carry) = lo.overflowing_add(t_lo);
    let (mut result, carry2) = hi.overflowing_add(t_hi);
    let (result2, carry3) = result.overflowing_add(carry as u64);
    result = result2;

    if carry2 || carry3 || result >= P {
        result = result.wrapping_sub(P);
    }

    result
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
            let result = from_mont(GoldilocksAvx512::scalar_add(to_mont(a), to_mont(b)));
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
            let result = from_mont(GoldilocksAvx512::scalar_sub(to_mont(a), to_mont(b)));
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn test_avx512_mont_mul() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a: [F64; 8] = core::array::from_fn(|_| F64::rand(&mut rng));
            let b: [F64; 8] = core::array::from_fn(|_| F64::rand(&mut rng));

            let a_raw: [u64; 8] = core::array::from_fn(|i| to_mont(a[i]));
            let b_raw: [u64; 8] = core::array::from_fn(|i| to_mont(b[i]));

            let a_v = unsafe { GoldilocksAvx512::load(a_raw.as_ptr()) };
            let b_v = unsafe { GoldilocksAvx512::load(b_raw.as_ptr()) };
            let r_v = GoldilocksAvx512::mul(a_v, b_v);

            let mut result = [0u64; 8];
            unsafe { GoldilocksAvx512::store(result.as_mut_ptr(), r_v) };

            for i in 0..8 {
                assert_eq!(from_mont(result[i]), a[i] * b[i], "lane {i} mul mismatch");
            }
        }
    }

    #[test]
    fn test_avx512_add() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a: [F64; 8] = core::array::from_fn(|_| F64::rand(&mut rng));
            let b: [F64; 8] = core::array::from_fn(|_| F64::rand(&mut rng));

            let a_raw: [u64; 8] = core::array::from_fn(|i| to_mont(a[i]));
            let b_raw: [u64; 8] = core::array::from_fn(|i| to_mont(b[i]));

            let a_v = unsafe { GoldilocksAvx512::load(a_raw.as_ptr()) };
            let b_v = unsafe { GoldilocksAvx512::load(b_raw.as_ptr()) };
            let r_v = GoldilocksAvx512::add(a_v, b_v);

            let mut result = [0u64; 8];
            unsafe { GoldilocksAvx512::store(result.as_mut_ptr(), r_v) };

            for i in 0..8 {
                assert_eq!(from_mont(result[i]), a[i] + b[i], "lane {i} add mismatch");
            }
        }
    }

    #[test]
    fn test_avx512_sub() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a: [F64; 8] = core::array::from_fn(|_| F64::rand(&mut rng));
            let b: [F64; 8] = core::array::from_fn(|_| F64::rand(&mut rng));

            let a_raw: [u64; 8] = core::array::from_fn(|i| to_mont(a[i]));
            let b_raw: [u64; 8] = core::array::from_fn(|i| to_mont(b[i]));

            let a_v = unsafe { GoldilocksAvx512::load(a_raw.as_ptr()) };
            let b_v = unsafe { GoldilocksAvx512::load(b_raw.as_ptr()) };
            let r_v = GoldilocksAvx512::sub(a_v, b_v);

            let mut result = [0u64; 8];
            unsafe { GoldilocksAvx512::store(result.as_mut_ptr(), r_v) };

            for i in 0..8 {
                assert_eq!(from_mont(result[i]), a[i] - b[i], "lane {i} sub mismatch");
            }
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
    fn test_avx512_edge_cases_vectorized() {
        use ark_ff::Field;
        let zero = F64::ZERO;
        let one = F64::ONE;
        let neg_one = -F64::ONE;

        // Test with all-zero, all-one, all-neg_one, and mixed lanes
        let a_vals = [zero, one, neg_one, one, zero, neg_one, one, neg_one];
        let b_vals = [neg_one, neg_one, neg_one, one, zero, one, zero, zero];
        let expected: [F64; 8] = core::array::from_fn(|i| a_vals[i] * b_vals[i]);

        let a_raw: [u64; 8] = core::array::from_fn(|i| to_mont(a_vals[i]));
        let b_raw: [u64; 8] = core::array::from_fn(|i| to_mont(b_vals[i]));

        let a_v = unsafe { GoldilocksAvx512::load(a_raw.as_ptr()) };
        let b_v = unsafe { GoldilocksAvx512::load(b_raw.as_ptr()) };
        let r_v = GoldilocksAvx512::mul(a_v, b_v);

        let mut result = [0u64; 8];
        unsafe { GoldilocksAvx512::store(result.as_mut_ptr(), r_v) };

        for i in 0..8 {
            assert_eq!(
                from_mont(result[i]),
                expected[i],
                "edge case lane {i} mismatch"
            );
        }
    }
}
