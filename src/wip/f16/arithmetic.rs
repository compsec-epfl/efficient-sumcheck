use ark_std::simd::{cmp::SimdPartialOrd, num::SimdInt, num::SimdUint, u16x4, Simd};

use crate::tests::SmallF16;

const F16_MODULUS: u16 = 65521;
const MONT_P_DASH: u16 = 0xEEEF;

#[inline(always)]
fn montgomery_mul_u16x4(a: Simd<u16, LANES>, b: Simd<u16, LANES>) -> Simd<u16, LANES> {
    // widen to u32 lanes for intermediate math
    let a32: Simd<u32, LANES> = a.cast();
    let b32: Simd<u32, LANES> = b.cast();

    let p32 = Simd::<u32, LANES>::splat(F16_MODULUS as u32);
    let p_dash32 = Simd::<u32, LANES>::splat(MONT_P_DASH as u32);
    let rmask32 = Simd::<u32, LANES>::splat(0xFFFF);

    // T = A * B (mod 2^32 via u32 wrapping)
    let t = a32 * b32;

    // m = (T * n') mod R  (only low 16 bits matter)
    let m = (t * p_dash32) & rmask32;

    // sum = T + m * p (mod 2^32)
    let mp = m * p32;
    let sum = t + mp;

    // carry = 1 if overflow in sum, else 0
    let carry_mask = sum.simd_lt(t); // if wrapped, sum < T
    let carry_ones: Simd<u32, LANES> = carry_mask.to_int().cast();
    let carry = carry_ones & Simd::<u32, LANES>::splat(1);

    // u = (sum >> 16) + (carry << 16)
    let hi = sum >> Simd::<u32, LANES>::splat(16);
    let carry_hi = carry << Simd::<u32, LANES>::splat(16);
    let mut u = hi + carry_hi;

    // if u >= p, subtract p (per lane)
    let ge_mask = u.simd_ge(p32);
    let u_minus_p = u - p32;
    u = ge_mask.select(u_minus_p, u);

    // back to u16 Montgomery repr
    u.cast()
}

const LANES: usize = 4;
pub fn mul_assign_16_bit_vectorized(a: &mut [u16], b: &[u16]) {
    let simd_end = a.len() / 16 * 16;
    for i in (0..simd_end).step_by(16) {
        let a_1: Simd<u16, LANES> = u16x4::from_slice(&a[i..i + 4]);
        let a_2: Simd<u16, LANES> = u16x4::from_slice(&a[i + 4..i + 8]);
        let a_3: Simd<u16, LANES> = u16x4::from_slice(&a[i + 8..i + 12]);
        let a_4: Simd<u16, LANES> = u16x4::from_slice(&a[i + 12..i + 16]);

        let b_1: Simd<u16, LANES> = u16x4::from_slice(&b[i..i + 4]);
        let b_2: Simd<u16, LANES> = u16x4::from_slice(&b[i + 4..i + 8]);
        let b_3: Simd<u16, LANES> = u16x4::from_slice(&b[i + 8..i + 12]);
        let b_4: Simd<u16, LANES> = u16x4::from_slice(&b[i + 12..i + 16]);

        let a_1_reduced = montgomery_mul_u16x4(a_1, b_1);
        let a_2_reduced = montgomery_mul_u16x4(a_2, b_2);
        let a_3_reduced = montgomery_mul_u16x4(a_3, b_3);
        let a_4_reduced = montgomery_mul_u16x4(a_4, b_4);

        a[i..i + 4].copy_from_slice(a_1_reduced.as_array());
        a[i + 4..i + 8].copy_from_slice(a_2_reduced.as_array());
        a[i + 8..i + 12].copy_from_slice(a_3_reduced.as_array());
        a[i + 12..i + 16].copy_from_slice(a_4_reduced.as_array());
    }

    // handle remainder
    let a_tail_f: &mut [SmallF16] = unsafe {
        core::slice::from_raw_parts_mut(
            a[simd_end..].as_mut_ptr() as *mut SmallF16,
            a.len() - simd_end,
        )
    };
    let b_tail_f: &[SmallF16] = unsafe {
        core::slice::from_raw_parts(
            b[simd_end..].as_ptr() as *const SmallF16,
            b.len() - simd_end,
        )
    };
    for (ai, bi) in a_tail_f.iter_mut().zip(b_tail_f.iter()) {
        *ai *= *bi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn vectorized_mul() {
        const LEN: usize = (1 << 20) + 3;
        let mut rng = test_rng();

        // we're going to modify these random vectors
        let mut expected: Vec<SmallF16> = (0..LEN).map(|_| SmallF16::rand(&mut rng)).collect();
        let mut received = expected.clone();
        let multiplicands: Vec<SmallF16> = (0..LEN).map(|_| SmallF16::rand(&mut rng)).collect();

        // control group
        for (a, b) in expected.iter_mut().zip(multiplicands.iter()) {
            *a *= b;
        }

        // test group
        let a_u16: &mut [u16] = unsafe {
            core::slice::from_raw_parts_mut(received.as_mut_ptr() as *mut u16, received.len())
        };
        let b_u16: &[u16] = unsafe {
            core::slice::from_raw_parts(multiplicands.as_ptr() as *const u16, multiplicands.len())
        };
        mul_assign_16_bit_vectorized(a_u16, b_u16);

        assert_eq!(expected, received);
    }
}
