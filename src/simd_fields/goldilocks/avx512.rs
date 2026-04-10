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

// ── Extension field arithmetic ──
//
// Extension field SIMD multiplication is not part of the SimdBaseField trait —
// it's implemented as free functions because the nonresidue `w` is a runtime
// value (extracted from the arkworks extension field config during dispatch).

/// Degree-2 Karatsuba: (a0 + a1·X)(b0 + b1·X) mod (X² - w)
/// 3 base muls + 1 mul-by-w + adds.
#[inline(always)]
pub fn ext2_mul(a: [__m512i; 2], b: [__m512i; 2], w: __m512i) -> [__m512i; 2] {
    let v0 = GoldilocksAvx512::mul(a[0], b[0]);
    let v1 = GoldilocksAvx512::mul(a[1], b[1]);
    let c0 = GoldilocksAvx512::add(v0, GoldilocksAvx512::mul(w, v1));
    let a_sum = GoldilocksAvx512::add(a[0], a[1]);
    let b_sum = GoldilocksAvx512::add(b[0], b[1]);
    let c1 = GoldilocksAvx512::sub(
        GoldilocksAvx512::sub(GoldilocksAvx512::mul(a_sum, b_sum), v0),
        v1,
    );
    [c0, c1]
}

/// Degree-2 Karatsuba (scalar version for tail processing).
#[inline(always)]
pub fn ext2_scalar_mul(a: [u64; 2], b: [u64; 2], w: u64) -> [u64; 2] {
    let v0 = mont_mul(a[0], b[0]);
    let v1 = mont_mul(a[1], b[1]);
    let c0 = GoldilocksAvx512::scalar_add(v0, mont_mul(w, v1));
    let a_sum = GoldilocksAvx512::scalar_add(a[0], a[1]);
    let b_sum = GoldilocksAvx512::scalar_add(b[0], b[1]);
    let c1 = GoldilocksAvx512::scalar_sub(
        GoldilocksAvx512::scalar_sub(mont_mul(a_sum, b_sum), v0),
        v1,
    );
    [c0, c1]
}

/// Degree-3 Karatsuba: (a0 + a1·X + a2·X²)(b0 + b1·X + b2·X²) mod (X³ - w)
/// 6 base muls + 2 mul-by-w + adds.
#[inline(always)]
pub fn ext3_mul(a: [__m512i; 3], b: [__m512i; 3], w: __m512i) -> [__m512i; 3] {
    let ad = GoldilocksAvx512::mul(a[0], b[0]);
    let be = GoldilocksAvx512::mul(a[1], b[1]);
    let cf = GoldilocksAvx512::mul(a[2], b[2]);

    let x = GoldilocksAvx512::sub(
        GoldilocksAvx512::sub(
            GoldilocksAvx512::mul(
                GoldilocksAvx512::add(a[1], a[2]),
                GoldilocksAvx512::add(b[1], b[2]),
            ),
            be,
        ),
        cf,
    );
    let y = GoldilocksAvx512::sub(
        GoldilocksAvx512::sub(
            GoldilocksAvx512::mul(
                GoldilocksAvx512::add(a[0], a[1]),
                GoldilocksAvx512::add(b[0], b[1]),
            ),
            ad,
        ),
        be,
    );
    let z = GoldilocksAvx512::add(
        GoldilocksAvx512::sub(
            GoldilocksAvx512::sub(
                GoldilocksAvx512::mul(
                    GoldilocksAvx512::add(a[0], a[2]),
                    GoldilocksAvx512::add(b[0], b[2]),
                ),
                ad,
            ),
            cf,
        ),
        be,
    );

    [
        GoldilocksAvx512::add(ad, GoldilocksAvx512::mul(w, x)),
        GoldilocksAvx512::add(y, GoldilocksAvx512::mul(w, cf)),
        z,
    ]
}

/// Degree-3 Karatsuba (scalar version).
#[inline(always)]
pub fn ext3_scalar_mul(a: [u64; 3], b: [u64; 3], w: u64) -> [u64; 3] {
    let ad = mont_mul(a[0], b[0]);
    let be = mont_mul(a[1], b[1]);
    let cf = mont_mul(a[2], b[2]);

    let x = GoldilocksAvx512::scalar_sub(
        GoldilocksAvx512::scalar_sub(
            mont_mul(
                GoldilocksAvx512::scalar_add(a[1], a[2]),
                GoldilocksAvx512::scalar_add(b[1], b[2]),
            ),
            be,
        ),
        cf,
    );
    let y = GoldilocksAvx512::scalar_sub(
        GoldilocksAvx512::scalar_sub(
            mont_mul(
                GoldilocksAvx512::scalar_add(a[0], a[1]),
                GoldilocksAvx512::scalar_add(b[0], b[1]),
            ),
            ad,
        ),
        be,
    );
    let z = GoldilocksAvx512::scalar_add(
        GoldilocksAvx512::scalar_sub(
            GoldilocksAvx512::scalar_sub(
                mont_mul(
                    GoldilocksAvx512::scalar_add(a[0], a[2]),
                    GoldilocksAvx512::scalar_add(b[0], b[2]),
                ),
                ad,
            ),
            cf,
        ),
        be,
    );

    [
        GoldilocksAvx512::scalar_add(ad, mont_mul(w, x)),
        GoldilocksAvx512::scalar_add(y, mont_mul(w, cf)),
        z,
    ]
}

/// Vectorized ext2 reduce: processes 8 pairs of degree-2 extension elements.
///
/// Input: 32 u64s in AoS layout: `[a0_c0, a0_c1, b0_c0, b0_c1, a1_c0, ...]`
/// Each group of 4 u64s is one pair `(a_i, b_i)` where a,b are ext2 elements.
/// Computes `result_i = a_i + challenge * (b_i - a_i)` for 8 pairs simultaneously.
/// Output: 16 u64s in AoS layout: `[r0_c0, r0_c1, r1_c0, r1_c1, ...]`
#[inline(always)]
pub unsafe fn ext2_reduce_8pairs(
    src: *const u64,
    dst: *mut u64,
    challenge_c0: __m512i,
    challenge_c1: __m512i,
    w_vec: __m512i,
) {
    // Load 32 u64s (4 cache lines worth)
    let v0 = _mm512_loadu_si512(src.cast());          // pairs 0-1: [a0c0,a0c1,b0c0,b0c1, a1c0,a1c1,b1c0,b1c1]
    let v1 = _mm512_loadu_si512(src.add(8).cast());   // pairs 2-3
    let v2 = _mm512_loadu_si512(src.add(16).cast());  // pairs 4-5
    let v3 = _mm512_loadu_si512(src.add(24).cast());  // pairs 6-7

    // Deinterleave: extract a_c0, a_c1, b_c0, b_c1 each as 8-wide vectors.
    // Within each 512-bit register, stride is 4: positions 0,4 are a_c0; 1,5 are a_c1; etc.
    // Across 4 registers: we gather element [k] from register [k/2], lane [4*(k%2) + component].
    //
    // a_c0: from (v0 lane 0), (v0 lane 4), (v1 lane 0), (v1 lane 4), (v2 lane 0), (v2 lane 4), (v3 lane 0), (v3 lane 4)
    // This requires cross-register shuffles. Use permutex2var for pairs of registers,
    // then a second round.

    // First round: extract even-pair and odd-pair components from adjacent register pairs
    // From v0,v1: gather a_c0 at indices 0,4 from v0 (=lanes 0,4) and 0,4 from v1 (=lanes 8,12)
    // permutex2var across v0,v1 gives us 8 values; we want the lower 4 from v0 and lower 4 from v1
    // permutex2var treats v0 as indices 0-7 and v1 as indices 8-15
    let a_c0_lo = _mm512_permutex2var_epi64(v0, _mm512_set_epi64(12, 8, 4, 0, 12, 8, 4, 0), v1);
    let a_c1_lo = _mm512_permutex2var_epi64(v0, _mm512_set_epi64(13, 9, 5, 1, 13, 9, 5, 1), v1);
    let b_c0_lo = _mm512_permutex2var_epi64(v0, _mm512_set_epi64(14, 10, 6, 2, 14, 10, 6, 2), v1);
    let b_c1_lo = _mm512_permutex2var_epi64(v0, _mm512_set_epi64(15, 11, 7, 3, 15, 11, 7, 3), v1);

    let a_c0_hi = _mm512_permutex2var_epi64(v2, _mm512_set_epi64(12, 8, 4, 0, 12, 8, 4, 0), v3);
    let a_c1_hi = _mm512_permutex2var_epi64(v2, _mm512_set_epi64(13, 9, 5, 1, 13, 9, 5, 1), v3);
    let b_c0_hi = _mm512_permutex2var_epi64(v2, _mm512_set_epi64(14, 10, 6, 2, 14, 10, 6, 2), v3);
    let b_c1_hi = _mm512_permutex2var_epi64(v2, _mm512_set_epi64(15, 11, 7, 3, 15, 11, 7, 3), v3);

    // Second round: merge lo (pairs 0-3 in lanes 0-3) and hi (pairs 4-7 in lanes 0-3)
    // into final 8-wide vectors.
    // lo has useful data in lanes 0-3, hi has useful data in lanes 0-3.
    // Use permutex2var: take lanes 0-3 from lo (indices 0-3) and lanes 0-3 from hi (indices 8-11).
    let idx_merge = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);

    let a_c0 = _mm512_permutex2var_epi64(a_c0_lo, idx_merge, a_c0_hi);
    let a_c1 = _mm512_permutex2var_epi64(a_c1_lo, idx_merge, a_c1_hi);
    let b_c0 = _mm512_permutex2var_epi64(b_c0_lo, idx_merge, b_c0_hi);
    let b_c1 = _mm512_permutex2var_epi64(b_c1_lo, idx_merge, b_c1_hi);

    // Compute diff = b - a (component-wise)
    let diff_c0 = GoldilocksAvx512::sub(b_c0, a_c0);
    let diff_c1 = GoldilocksAvx512::sub(b_c1, a_c1);

    // prod = challenge * diff (ext2 Karatsuba)
    let prod = ext2_mul([diff_c0, diff_c1], [challenge_c0, challenge_c1], w_vec);

    // result = a + prod
    let r_c0 = GoldilocksAvx512::add(a_c0, prod[0]);
    let r_c1 = GoldilocksAvx512::add(a_c1, prod[1]);

    // Interleave back to AoS: [r0_c0, r0_c1, r1_c0, r1_c1, ...]
    // 8 results → 16 u64s in 2 registers
    // r_c0 = [r0, r1, r2, r3, r4, r5, r6, r7] (component 0)
    // r_c1 = [r0, r1, r2, r3, r4, r5, r6, r7] (component 1)
    // Want: out0 = [r0c0,r0c1,r1c0,r1c1,r2c0,r2c1,r3c0,r3c1]
    //       out1 = [r4c0,r4c1,r5c0,r5c1,r6c0,r6c1,r7c0,r7c1]
    let idx_interleave_lo = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
    let idx_interleave_hi = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
    let out0 = _mm512_permutex2var_epi64(r_c0, idx_interleave_lo, r_c1);
    let out1 = _mm512_permutex2var_epi64(r_c0, idx_interleave_hi, r_c1);

    _mm512_storeu_si512(dst.cast(), out0);
    _mm512_storeu_si512(dst.add(8).cast(), out1);
}

/// Vectorized ext3 reduce: processes 8 pairs of degree-3 extension elements.
///
/// Input: 48 u64s in AoS layout: `[a0_c0, a0_c1, a0_c2, b0_c0, b0_c1, b0_c2, a1_c0, ...]`
/// Each group of 6 u64s is one pair `(a_i, b_i)` where a,b are ext3 elements.
/// Computes `result_i = a_i + challenge * (b_i - a_i)` for 8 pairs simultaneously.
/// Output: 24 u64s in AoS layout: `[r0_c0, r0_c1, r0_c2, r1_c0, r1_c1, r1_c2, ...]`
///
/// Uses AVX-512 gather/scatter for the stride-6 deinterleave/interleave.
#[inline(always)]
pub unsafe fn ext3_reduce_8pairs(
    src: *const u64,
    dst: *mut u64,
    challenge: [__m512i; 3],
    w_vec: __m512i,
) {
    // Gather 6 components from AoS layout (stride 6 per pair)
    // Pair i: a at offset 6i, b at offset 6i+3
    let idx_a_c0 = _mm512_set_epi64(42, 36, 30, 24, 18, 12, 6, 0);
    let idx_a_c1 = _mm512_set_epi64(43, 37, 31, 25, 19, 13, 7, 1);
    let idx_a_c2 = _mm512_set_epi64(44, 38, 32, 26, 20, 14, 8, 2);
    let idx_b_c0 = _mm512_set_epi64(45, 39, 33, 27, 21, 15, 9, 3);
    let idx_b_c1 = _mm512_set_epi64(46, 40, 34, 28, 22, 16, 10, 4);
    let idx_b_c2 = _mm512_set_epi64(47, 41, 35, 29, 23, 17, 11, 5);

    let base = src as *const i64;
    let a_c0 = _mm512_i64gather_epi64::<8>(idx_a_c0, base);
    let a_c1 = _mm512_i64gather_epi64::<8>(idx_a_c1, base);
    let a_c2 = _mm512_i64gather_epi64::<8>(idx_a_c2, base);
    let b_c0 = _mm512_i64gather_epi64::<8>(idx_b_c0, base);
    let b_c1 = _mm512_i64gather_epi64::<8>(idx_b_c1, base);
    let b_c2 = _mm512_i64gather_epi64::<8>(idx_b_c2, base);

    // diff = b - a (component-wise)
    let diff_c0 = GoldilocksAvx512::sub(b_c0, a_c0);
    let diff_c1 = GoldilocksAvx512::sub(b_c1, a_c1);
    let diff_c2 = GoldilocksAvx512::sub(b_c2, a_c2);

    // prod = challenge * diff (ext3 Karatsuba)
    let prod = ext3_mul([diff_c0, diff_c1, diff_c2], challenge, w_vec);

    // result = a + prod
    let r_c0 = GoldilocksAvx512::add(a_c0, prod[0]);
    let r_c1 = GoldilocksAvx512::add(a_c1, prod[1]);
    let r_c2 = GoldilocksAvx512::add(a_c2, prod[2]);

    // Scatter back to AoS (stride 3 per result element)
    let idx_r_c0 = _mm512_set_epi64(21, 18, 15, 12, 9, 6, 3, 0);
    let idx_r_c1 = _mm512_set_epi64(22, 19, 16, 13, 10, 7, 4, 1);
    let idx_r_c2 = _mm512_set_epi64(23, 20, 17, 14, 11, 8, 5, 2);

    let base_out = dst as *mut i64;
    _mm512_i64scatter_epi64::<8>(base_out, idx_r_c0, r_c0);
    _mm512_i64scatter_epi64::<8>(base_out, idx_r_c1, r_c1);
    _mm512_i64scatter_epi64::<8>(base_out, idx_r_c2, r_c2);
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

    #[test]
    fn test_ext2_scalar_mul() {
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
            let expected_c0 = a0 * b0 + w * (a1 * b2 + a2 * b1);
            let expected_c1 = a0 * b1 + a1 * b0 + w * a2 * b2;
            let expected_c2 = a0 * b2 + a1 * b1 + a2 * b0;

            assert_eq!(from_mont(result[0]), expected_c0, "ext3 c0 mismatch");
            assert_eq!(from_mont(result[1]), expected_c1, "ext3 c1 mismatch");
            assert_eq!(from_mont(result[2]), expected_c2, "ext3 c2 mismatch");
        }
    }

    #[test]
    fn test_ext2_avx512_matches_scalar() {
        let mut rng = test_rng();
        let w_mont = to_mont(F64::from(7u64));
        let w_vec = GoldilocksAvx512::splat(w_mont);

        for _ in 0..10_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);

            // Broadcast same values across all 8 lanes
            let a_v = [
                GoldilocksAvx512::splat(to_mont(a0)),
                GoldilocksAvx512::splat(to_mont(a1)),
            ];
            let b_v = [
                GoldilocksAvx512::splat(to_mont(b0)),
                GoldilocksAvx512::splat(to_mont(b1)),
            ];

            let r_v = ext2_mul(a_v, b_v, w_vec);

            let mut r_out = [[0u64; 8]; 2];
            unsafe {
                GoldilocksAvx512::store(r_out[0].as_mut_ptr(), r_v[0]);
                GoldilocksAvx512::store(r_out[1].as_mut_ptr(), r_v[1]);
            }

            let scalar_result = ext2_scalar_mul(
                [to_mont(a0), to_mont(a1)],
                [to_mont(b0), to_mont(b1)],
                w_mont,
            );

            for lane in 0..8 {
                assert_eq!(
                    r_out[0][lane], scalar_result[0],
                    "ext2 AVX-512 c0 lane {lane} mismatch"
                );
                assert_eq!(
                    r_out[1][lane], scalar_result[1],
                    "ext2 AVX-512 c1 lane {lane} mismatch"
                );
            }
        }
    }

    #[test]
    fn test_ext3_avx512_matches_scalar() {
        let mut rng = test_rng();
        let w_mont = to_mont(F64::from(7u64));
        let w_vec = GoldilocksAvx512::splat(w_mont);

        for _ in 0..10_000 {
            let a0 = F64::rand(&mut rng);
            let a1 = F64::rand(&mut rng);
            let a2 = F64::rand(&mut rng);
            let b0 = F64::rand(&mut rng);
            let b1 = F64::rand(&mut rng);
            let b2 = F64::rand(&mut rng);

            let a_v = [
                GoldilocksAvx512::splat(to_mont(a0)),
                GoldilocksAvx512::splat(to_mont(a1)),
                GoldilocksAvx512::splat(to_mont(a2)),
            ];
            let b_v = [
                GoldilocksAvx512::splat(to_mont(b0)),
                GoldilocksAvx512::splat(to_mont(b1)),
                GoldilocksAvx512::splat(to_mont(b2)),
            ];

            let r_v = ext3_mul(a_v, b_v, w_vec);

            let mut r_out = [[0u64; 8]; 3];
            unsafe {
                GoldilocksAvx512::store(r_out[0].as_mut_ptr(), r_v[0]);
                GoldilocksAvx512::store(r_out[1].as_mut_ptr(), r_v[1]);
                GoldilocksAvx512::store(r_out[2].as_mut_ptr(), r_v[2]);
            }

            let scalar_result = ext3_scalar_mul(
                [to_mont(a0), to_mont(a1), to_mont(a2)],
                [to_mont(b0), to_mont(b1), to_mont(b2)],
                w_mont,
            );

            for lane in 0..8 {
                assert_eq!(
                    r_out[0][lane], scalar_result[0],
                    "ext3 AVX-512 c0 lane {lane} mismatch"
                );
                assert_eq!(
                    r_out[1][lane], scalar_result[1],
                    "ext3 AVX-512 c1 lane {lane} mismatch"
                );
                assert_eq!(
                    r_out[2][lane], scalar_result[2],
                    "ext3 AVX-512 c2 lane {lane} mismatch"
                );
            }
        }
    }

    #[test]
    fn test_ext2_reduce_8pairs() {
        let mut rng = test_rng();
        let w_mont = to_mont(F64::from(7u64));

        for _ in 0..1_000 {
            // Generate 8 pairs of ext2 elements in AoS layout (32 u64s)
            let src: Vec<u64> = (0..32).map(|_| to_mont(F64::rand(&mut rng))).collect();
            let challenge = [to_mont(F64::rand(&mut rng)), to_mont(F64::rand(&mut rng))];

            // Reference: scalar reduce
            let mut expected = vec![0u64; 16];
            for i in 0..8 {
                let a = [src[4 * i], src[4 * i + 1]];
                let b = [src[4 * i + 2], src[4 * i + 3]];
                let diff = [
                    GoldilocksAvx512::scalar_sub(b[0], a[0]),
                    GoldilocksAvx512::scalar_sub(b[1], a[1]),
                ];
                let prod = ext2_scalar_mul(diff, challenge, w_mont);
                expected[2 * i] = GoldilocksAvx512::scalar_add(a[0], prod[0]);
                expected[2 * i + 1] = GoldilocksAvx512::scalar_add(a[1], prod[1]);
            }

            // Vectorized
            let mut actual = vec![0u64; 16];
            let challenge_c0 = GoldilocksAvx512::splat(challenge[0]);
            let challenge_c1 = GoldilocksAvx512::splat(challenge[1]);
            let w_vec = GoldilocksAvx512::splat(w_mont);
            unsafe {
                ext2_reduce_8pairs(
                    src.as_ptr(),
                    actual.as_mut_ptr(),
                    challenge_c0,
                    challenge_c1,
                    w_vec,
                );
            }

            assert_eq!(expected, actual, "ext2_reduce_8pairs mismatch");
        }
    }

    #[test]
    fn test_ext3_reduce_8pairs() {
        let mut rng = test_rng();
        let w_mont = to_mont(F64::from(7u64));

        for _ in 0..1_000 {
            // Generate 8 pairs of ext3 elements in AoS layout (48 u64s)
            let src: Vec<u64> = (0..48).map(|_| to_mont(F64::rand(&mut rng))).collect();
            let challenge = [
                to_mont(F64::rand(&mut rng)),
                to_mont(F64::rand(&mut rng)),
                to_mont(F64::rand(&mut rng)),
            ];

            // Reference: scalar reduce
            let mut expected = vec![0u64; 24];
            for i in 0..8 {
                let a = [src[6 * i], src[6 * i + 1], src[6 * i + 2]];
                let b = [src[6 * i + 3], src[6 * i + 4], src[6 * i + 5]];
                let diff = [
                    GoldilocksAvx512::scalar_sub(b[0], a[0]),
                    GoldilocksAvx512::scalar_sub(b[1], a[1]),
                    GoldilocksAvx512::scalar_sub(b[2], a[2]),
                ];
                let prod = ext3_scalar_mul(diff, challenge, w_mont);
                expected[3 * i] = GoldilocksAvx512::scalar_add(a[0], prod[0]);
                expected[3 * i + 1] = GoldilocksAvx512::scalar_add(a[1], prod[1]);
                expected[3 * i + 2] = GoldilocksAvx512::scalar_add(a[2], prod[2]);
            }

            // Vectorized
            let mut actual = vec![0u64; 24];
            let challenge_v = [
                GoldilocksAvx512::splat(challenge[0]),
                GoldilocksAvx512::splat(challenge[1]),
                GoldilocksAvx512::splat(challenge[2]),
            ];
            let w_vec = GoldilocksAvx512::splat(w_mont);
            unsafe {
                ext3_reduce_8pairs(
                    src.as_ptr(),
                    actual.as_mut_ptr(),
                    challenge_v,
                    w_vec,
                );
            }

            assert_eq!(expected, actual, "ext3_reduce_8pairs mismatch");
        }
    }
}
