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

use crate::tests::{Fp4SmallM31, SmallM31};

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

pub fn reduce_evaluations_bf(src: &[SmallM31], verifier_message: Fp4SmallM31) -> Vec<Fp4SmallM31> {
    cfg_chunks!(src, 2)
        .map(|chunk| {
            let v0 = Fp4SmallM31::from_base_prime_field(chunk[0]);
            let v1 = Fp4SmallM31::from_base_prime_field(chunk[1]);
            v0 + verifier_message * (v1 - v0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use crate::tests::Fp4SmallM31;
    use crate::{multilinear::pairwise, tests::SmallM31};

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
}
