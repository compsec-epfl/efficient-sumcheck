use ark_std::simd::{cmp::SimdPartialOrd, num::SimdUint, u32x4, Simd};

use crate::tests::SmallM31;

const M31_MODULUS: u32 = 2_147_483_647;

const LANES: usize = 4;

// #[inline(always)]
// pub fn mul_mod_m31_u32x4(a: Simd<u32, LANES>, b: Simd<u32, LANES>) -> Simd<u32, LANES> {
//     let a64: Simd<u64, LANES> = a.cast();
//     let b64: Simd<u64, LANES> = b.cast();
//     let t = a64 * b64;

//     let mask = Simd::<u64, LANES>::splat((1u64 << 31) - 1);
//     let p64 = Simd::<u64, LANES>::splat(M31_MODULUS as u64);

//     // Mersenne reduction
//     let low = t & mask;
//     let high = t >> Simd::<u64, LANES>::splat(31);
//     let mut x = low + high;

//     // At most 2 subtractions needed
//     let ge1 = x.simd_ge(p64);
//     x = ge1.select(x - p64, x);

//     let ge2 = x.simd_ge(p64);
//     x = ge2.select(x - p64, x);

//     x.cast()
// }

#[inline(always)]
pub fn sub_mod_m31_u32x4(a: Simd<u32, LANES>, b: Simd<u32, LANES>) -> Simd<u32, LANES> {
    // tmp = a - b (mod 2^32)
    let tmp = a - b;

    // detect wraparound: a < b  (per-lane borrow)
    let borrow = a.simd_lt(b);

    // if wrapped, add modulus back; else keep tmp
    let p = Simd::<u32, LANES>::splat(M31_MODULUS);
    let tmp_plus_p = tmp + p;

    borrow.select(tmp_plus_p, tmp)
}

#[inline(always)]
pub fn add_mod_m31_u32x4(a: Simd<u32, LANES>, b: Simd<u32, LANES>) -> Simd<u32, LANES> {
    // lane-wise sum in u32
    let sum = a + b;

    // single reduction step: because 0 <= a,b < p, we have 0 <= sum <= 2p-2,
    // so at most one subtraction of p is needed
    let p = Simd::<u32, LANES>::splat(M31_MODULUS);
    let ge = sum.simd_ge(p); // mask: sum >= p
    ge.select(sum - p, sum) // if ge: sum - p else sum
}

pub fn mul_assign_m31_vectorized(a: &mut [u32], b: &[u32]) {
    assert_eq!(a.len(), b.len());

    let simd_end = a.len() / 16 * 16;

    // SIMD bulk: 16 elements at a time (4 x u32x4)
    for i in (0..simd_end).step_by(16) {
        let a_1: Simd<u32, LANES> = u32x4::from_slice(&a[i..i + 4]);
        let a_2: Simd<u32, LANES> = u32x4::from_slice(&a[i + 4..i + 8]);
        let a_3: Simd<u32, LANES> = u32x4::from_slice(&a[i + 8..i + 12]);
        let a_4: Simd<u32, LANES> = u32x4::from_slice(&a[i + 12..i + 16]);

        let b_1: Simd<u32, LANES> = u32x4::from_slice(&b[i..i + 4]);
        let b_2: Simd<u32, LANES> = u32x4::from_slice(&b[i + 4..i + 8]);
        let b_3: Simd<u32, LANES> = u32x4::from_slice(&b[i + 8..i + 12]);
        let b_4: Simd<u32, LANES> = u32x4::from_slice(&b[i + 12..i + 16]);

        let r1 = mul_mod_m31_u32x4(a_1, b_1);
        let r2 = mul_mod_m31_u32x4(a_2, b_2);
        let r3 = mul_mod_m31_u32x4(a_3, b_3);
        let r4 = mul_mod_m31_u32x4(a_4, b_4);

        a[i..i + 4].copy_from_slice(r1.as_array());
        a[i + 4..i + 8].copy_from_slice(r2.as_array());
        a[i + 8..i + 12].copy_from_slice(r3.as_array());
        a[i + 12..i + 16].copy_from_slice(r4.as_array());
    }

    // Scalar tail: reinterpret remainder as SmallM31 and do normal field mul.
    let a_tail_f: &mut [SmallM31] = unsafe {
        core::slice::from_raw_parts_mut(
            a[simd_end..].as_mut_ptr() as *mut SmallM31,
            a.len() - simd_end,
        )
    };
    let b_tail_f: &[SmallM31] = unsafe {
        core::slice::from_raw_parts(
            b[simd_end..].as_ptr() as *const SmallM31,
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

        let mut expected: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
        let mut received = expected.clone();
        let multiplicands: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();

        // scalar baseline
        for (a, b) in expected.iter_mut().zip(multiplicands.iter()) {
            *a *= b;
        }

        // SIMD version on the underlying u32 words
        let a_u32: &mut [u32] = unsafe {
            core::slice::from_raw_parts_mut(received.as_mut_ptr() as *mut u32, received.len())
        };
        let b_u32: &[u32] = unsafe {
            core::slice::from_raw_parts(multiplicands.as_ptr() as *const u32, multiplicands.len())
        };

        mul_assign_m31_vectorized(a_u32, b_u32);

        assert_eq!(expected, received);
    }
}
