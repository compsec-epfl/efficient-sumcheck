use ark_ff::Field;
use ark_std::simd::{cmp::SimdPartialOrd, num::SimdUint, u32x4, Mask, Simd};
use ark_std::{cfg_chunks, cfg_into_iter};
use ark_std::{mem, vec::Vec};
use rayon::current_num_threads;
#[cfg(feature = "parallel")]
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    prelude::ParallelSlice,
};

use crate::wip::m31::arithmetic::mul_mod_m31_u32x4;

use crate::tests::{Fp2SmallM31, Fp4SmallM31, SmallM31};

const M31_MODULUS: u32 = 2_147_483_647;

#[inline(always)]
fn sum_indexwise(a1: [u32; 4], a2: [u32; 4], a3: [u32; 4], a4: [u32; 4]) -> (u32, u32) {
    let mut even_sum = 0u32;
    let mut odd_sum = 0u32;

    let add_mod = |acc: &mut u32, x: u32| {
        let tmp = *acc + x;
        *acc = if tmp >= M31_MODULUS {
            tmp - M31_MODULUS
        } else {
            tmp
        };
    };

    for arr in [&a1, &a2, &a3, &a4] {
        add_mod(&mut even_sum, arr[0]);
        add_mod(&mut even_sum, arr[2]);

        add_mod(&mut odd_sum, arr[1]);
        add_mod(&mut odd_sum, arr[3]);
    }

    (even_sum, odd_sum)
}

#[inline(always)]
fn add_mod_val(a: u32, b: u32) -> u32 {
    let tmp = a + b;
    if tmp >= M31_MODULUS {
        tmp - M31_MODULUS
    } else {
        tmp
    }
}

#[inline(always)]
fn sub_mod_val(a: u32, b: u32) -> u32 {
    // We compute (a - b) mod M31.
    // If underflow happens (a < b), we add M31 back.
    // Wrapping subtract gives the correct limb but with wrap-around.
    let tmp = a.wrapping_sub(b);

    // If a < b, tmp wrapped around and is > a.
    // In that case, add modulus back.
    if a < b {
        tmp.wrapping_add(M31_MODULUS)
    } else {
        tmp
    }
}

const LANES: usize = 4;
pub fn reduce_sum_packed(values: &[u32]) -> (u32, u32) {
    let packed_modulus: Simd<u32, LANES> = u32x4::splat(M31_MODULUS);
    let mut packed_sums1: Simd<u32, LANES> = u32x4::splat(0);
    let mut packed_sums2: Simd<u32, LANES> = u32x4::splat(0);
    let mut packed_sums3: Simd<u32, LANES> = u32x4::splat(0);
    let mut packed_sums4: Simd<u32, LANES> = u32x4::splat(0);
    for i in (0..values.len()).step_by(16) {
        let tmp_packed_sums_1: Simd<u32, LANES> =
            packed_sums1 + u32x4::from_slice(&values[i..i + 4]);
        let tmp_packed_sums_2: Simd<u32, LANES> =
            packed_sums2 + u32x4::from_slice(&values[i + 4..i + 8]);
        let tmp_packed_sums_3: Simd<u32, LANES> =
            packed_sums3 + u32x4::from_slice(&values[i + 8..i + 12]);
        let tmp_packed_sums_4: Simd<u32, LANES> =
            packed_sums4 + u32x4::from_slice(&values[i + 12..i + 16]);
        let is_mod_needed_1: Mask<i32, LANES> = tmp_packed_sums_1.simd_ge(packed_modulus);
        let is_mod_needed_2: Mask<i32, LANES> = tmp_packed_sums_2.simd_ge(packed_modulus);
        let is_mod_needed_3: Mask<i32, LANES> = tmp_packed_sums_3.simd_ge(packed_modulus);
        let is_mod_needed_4: Mask<i32, LANES> = tmp_packed_sums_4.simd_ge(packed_modulus);
        packed_sums1 =
            is_mod_needed_1.select(tmp_packed_sums_1 - packed_modulus, tmp_packed_sums_1);
        packed_sums2 =
            is_mod_needed_2.select(tmp_packed_sums_2 - packed_modulus, tmp_packed_sums_2);
        packed_sums3 =
            is_mod_needed_3.select(tmp_packed_sums_3 - packed_modulus, tmp_packed_sums_3);
        packed_sums4 =
            is_mod_needed_4.select(tmp_packed_sums_4 - packed_modulus, tmp_packed_sums_4);
    }
    sum_indexwise(
        packed_sums1.to_array(),
        packed_sums2.to_array(),
        packed_sums3.to_array(),
        packed_sums4.to_array(),
    )
}

pub fn evaluate_bf(src: &[SmallM31]) -> (SmallM31, SmallM31) {
    assert!(src.len().is_multiple_of(16));

    // TODO (z-tech): this is machine dependent
    const PARALLEL_BREAK_EVEN: usize = 1 << 17;
    if src.len() < PARALLEL_BREAK_EVEN || !cfg!(feature = "parallel") {
        let a_raw: &[u32] =
            unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u32, src.len()) };
        let (sum_0_u32, sum_1_u32) = reduce_sum_packed(a_raw);
        let sum_0: SmallM31 = unsafe { mem::transmute::<u32, SmallM31>(sum_0_u32) };
        let sum_1: SmallM31 = unsafe { mem::transmute::<u32, SmallM31>(sum_1_u32) };
        return (sum_0, sum_1);
    }

    #[cfg(feature = "parallel")]
    {
        let n_threads = current_num_threads();
        let chunk_size = src.len() / n_threads;
        let sums = src
            .par_chunks(chunk_size) // minimum block = SIMD width
            .map(|chunk_of_16| {
                let chunk_of_16_raw: &[u32] = unsafe {
                    core::slice::from_raw_parts(
                        chunk_of_16.as_ptr() as *const u32,
                        chunk_of_16.len(),
                    )
                };
                reduce_sum_packed(chunk_of_16_raw)
            })
            .reduce(
                || (0, 0),
                |(e1, o1), (e2, o2)| (add_mod_val(e1, e2), add_mod_val(o1, o2)),
            );
        let (sum_0_u32, sum_1_u32) = sums;
        let sum_0: SmallM31 = unsafe { mem::transmute::<u32, SmallM31>(sum_0_u32) };
        let sum_1: SmallM31 = unsafe { mem::transmute::<u32, SmallM31>(sum_1_u32) };
        (sum_0, sum_1)
    }
}

#[inline(always)]
fn special_add(a: &SmallM31, b: &mut Fp4SmallM31) {
    let a_raw = unsafe { mem::transmute::<SmallM31, u32>(*a) };
    let limbs: &mut [u32; 4] = unsafe { &mut *(b as *mut Fp4SmallM31 as *mut [u32; 4]) };
    limbs[0] = add_mod_val(limbs[0], a_raw);
}

#[inline(always)]
fn special_mult(a: &SmallM31, b: &Fp4SmallM31) -> Fp4SmallM31 {
    let a_raw = unsafe { mem::transmute::<SmallM31, u32>(*a) };
    let a_packed: Simd<u32, 4> = Simd::<u32, 4>::splat(a_raw);
    let b_raw: [u32; 4] = unsafe {
        debug_assert_eq!(mem::size_of::<Fp4SmallM31>(), 4 * mem::size_of::<u32>());
        debug_assert_eq!(mem::align_of::<Fp4SmallM31>(), mem::align_of::<u32>());
        mem::transmute::<Fp4SmallM31, [u32; 4]>(*b)
    };
    let b_packed: Simd<u32, 4> = Simd::from_array(b_raw);
    let res = mul_mod_m31_u32x4(a_packed, b_packed);
    let res_raw: &[u32; 4] = res.as_array();
    unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(*res_raw) }
}

#[inline(always)]
fn special_thing(verifier_challenge: &Fp4SmallM31, v0: &SmallM31, v1: &SmallM31) -> Fp4SmallM31 {
    let v1_minus_v0 = v1 - v0;
    let v1_minus_v0_raw = unsafe { mem::transmute::<SmallM31, u32>(v1_minus_v0) };
    let v1_minus_v0_packed: Simd<u32, 4> = Simd::<u32, 4>::splat(v1_minus_v0_raw);
    let verifier_challenge_raw: [u32; 4] = unsafe {
        debug_assert_eq!(mem::size_of::<Fp4SmallM31>(), 4 * mem::size_of::<u32>());
        debug_assert_eq!(mem::align_of::<Fp4SmallM31>(), mem::align_of::<u32>());
        mem::transmute::<Fp4SmallM31, [u32; 4]>(*verifier_challenge)
    };
    let verifier_challenge_packed: Simd<u32, 4> = Simd::from_array(verifier_challenge_raw);
    let res = mul_mod_m31_u32x4(v1_minus_v0_packed, verifier_challenge_packed);
    let mut res_raw: [u32; 4] = *res.as_array();
    let v0_raw = unsafe { mem::transmute::<SmallM31, u32>(*v0) };
    res_raw[0] = add_mod_val(res_raw[0], v0_raw);
    unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(res_raw) }
}

pub fn special_thing_unrolled(verifier_challenge: &[u32; 4], src: &[u32]) -> Vec<[u32; 4]> {
    assert!(src.len().is_multiple_of(4));

    let verifier_challenge_packed: Simd<u32, 4> = Simd::from_array(*verifier_challenge);
    let mut out: Vec<[u32; 4]> = Vec::with_capacity(src.len() / 2);
    for i in (0..src.len()).step_by(8) {
        let r0 = mul_mod_m31_u32x4(
            verifier_challenge_packed,
            Simd::<u32, 4>::splat(sub_mod_val(src[i + 1], src[i])),
        );
        let mut r0_raw: [u32; 4] = *r0.as_array();
        r0_raw[0] = add_mod_val(r0_raw[0], src[i]);
        out.push(r0_raw);

        let r1 = mul_mod_m31_u32x4(
            verifier_challenge_packed,
            Simd::<u32, 4>::splat(sub_mod_val(src[i + 3], src[i + 2])),
        );
        let mut r1_raw: [u32; 4] = *r1.as_array();
        r1_raw[0] = add_mod_val(r1_raw[0], src[i + 2]);
        out.push(r1_raw);

        let r2 = mul_mod_m31_u32x4(
            verifier_challenge_packed,
            Simd::<u32, 4>::splat(sub_mod_val(src[i + 5], src[i + 4])),
        );
        let mut r2_raw: [u32; 4] = *r2.as_array();
        r2_raw[0] = add_mod_val(r2_raw[0], src[i + 4]);
        out.push(r2_raw);

        let r3 = mul_mod_m31_u32x4(
            verifier_challenge_packed,
            Simd::<u32, 4>::splat(sub_mod_val(src[i + 7], src[i + 6])),
        );
        let mut r3_raw: [u32; 4] = *r3.as_array();
        r3_raw[0] = add_mod_val(r3_raw[0], src[i + 6]);
        out.push(r3_raw);
    }
    out
}

pub fn reduce_evaluations_bf(src: &[SmallM31], verifier_message: Fp4SmallM31) -> Vec<Fp4SmallM31> {
    cfg_chunks!(src, 2)
        .map(|chunk| special_thing(&verifier_message, &chunk[0], &chunk[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::mem;

    use ark_ff::Field;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use crate::multilinear::pairwise;
    use crate::tests::{Fp2SmallM31, Fp4SmallM31, SmallM31};

    use crate::wip::m31::vectorized_reductions::pairwise::special_add;
    use crate::wip::m31::vectorized_reductions::pairwise::special_mult;
    use crate::wip::m31::vectorized_reductions::pairwise::special_thing;
    use crate::wip::m31::vectorized_reductions::pairwise::special_thing_unrolled;
    use crate::wip::m31::vectorized_reductions::pairwise::{evaluate_bf, reduce_evaluations_bf};

    #[test]
    fn sanity_evaluate_bf() {
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();
        let src: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
        let expected = pairwise::evaluate(&src);
        let received = evaluate_bf(&src);
        assert_eq!(expected, received);
    }

    #[test]
    fn sanity_reduce_evaluations_bf() {
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();
        let mut src: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
        let src_copy: Vec<SmallM31> = src.clone();

        let challenge = SmallM31::from(7);
        let challenge_ext = Fp4SmallM31::from_base_prime_field(challenge);

        pairwise::reduce_evaluations(&mut src, challenge);
        let expected: Vec<Fp4SmallM31> = src
            .into_iter()
            .map(Fp4SmallM31::from_base_prime_field)
            .collect();
        let received = reduce_evaluations_bf(&src_copy, challenge_ext);
        assert_eq!(expected, received);
    }

    #[test]
    fn sanity_special_mult() {
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();
        let base: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
        let ext: Vec<Fp4SmallM31> = (0..LEN).map(|_| Fp4SmallM31::rand(&mut rng)).collect();
        for (a, b) in base.iter().zip(ext.iter()) {
            let mut expected: Fp4SmallM31 = *b;
            expected.mul_assign_by_basefield(&Fp2SmallM31::from_base_prime_field(*a));
            let received = special_mult(a, b);
            assert_eq!(expected, received);
        }
    }

    #[test]
    fn sanity_special_add() {
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();
        let base: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
        let ext: Vec<Fp4SmallM31> = (0..LEN).map(|_| Fp4SmallM31::rand(&mut rng)).collect();
        for (a, b) in base.iter().zip(ext.iter()) {
            let a_ext = Fp4SmallM31::from_base_prime_field(*a);
            let expected: Fp4SmallM31 = *b + a_ext;
            let mut received = *b;
            special_add(a, &mut received);
            assert_eq!(expected, received);
        }
    }

    #[test]
    fn special_thing_vectorized_matches_scalar_single_vec() {
        let mut rng = test_rng();

        // Try several even lengths (since we pair as (v[2i], v[2i+1]))
        for &len in &[8, 16, 32, 64, 128] {
            assert!(len % 2 == 0);
            let verifier_challenge = Fp4SmallM31::rand(&mut rng);
            let v: Vec<SmallM31> = (0..len).map(|_| SmallM31::rand(&mut rng)).collect();

            let n_pairs = len / 2;

            // Scalar: apply special_thing on (v[2i], v[2i+1])
            let mut expected = Vec::with_capacity(n_pairs);
            for k in 0..n_pairs {
                let i0 = 2 * k;
                let i1 = i0 + 1;
                expected.push(special_thing(&verifier_challenge, &v[i0], &v[i1]));
            }

            // Vectorized version
            let verifier_message_raw: [u32; 4] = unsafe {
                debug_assert_eq!(mem::size_of::<Fp4SmallM31>(), 4 * mem::size_of::<u32>());
                debug_assert_eq!(mem::align_of::<Fp4SmallM31>(), mem::align_of::<u32>());
                mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_challenge)
            };

            let chunk_raw: &[u32] =
                unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u32, v.len()) };
            let raw_vec = special_thing_unrolled(&verifier_message_raw, chunk_raw);
            let ptr = raw_vec.as_ptr() as *mut Fp4SmallM31;
            let len = raw_vec.len();
            let cap = raw_vec.capacity();

            // take ownership of the buffer without freeing raw_vec's allocation
            core::mem::forget(raw_vec);

            let received = unsafe { Vec::from_raw_parts(ptr, len, cap) };

            assert_eq!(expected, received, "mismatch for len={len}");
        }
    }
}
