use ark_std::{
    cfg_into_iter, mem,
    simd::{cmp::SimdPartialOrd, num::SimdUint, Simd},
};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    tests::{Fp4SmallM31, SmallM31},
    wip::m31::evaluate_bf::add_mod_val,
};

#[inline(always)]
pub fn mul_mod_m31_u32x4<const MODULUS: u32>(a: Simd<u32, 4>, b: Simd<u32, 4>) -> Simd<u32, 4> {
    let a64: Simd<u64, 4> = a.cast();
    let b64: Simd<u64, 4> = b.cast();
    let t = a64 * b64;

    let mask = Simd::<u64, 4>::splat((1u64 << 31) - 1);
    let p64 = Simd::<u64, 4>::splat(MODULUS as u64);

    // Mersenne reduction
    let low = t & mask;
    let high = t >> Simd::<u64, 4>::splat(31);
    let mut x = low + high;

    // At most 2 subtractions needed
    let ge1 = x.simd_ge(p64);
    x = ge1.select(x - p64, x);

    let ge2 = x.simd_ge(p64);
    x = ge2.select(x - p64, x);

    x.cast()
}

pub fn reduce_bf(src: &[SmallM31], verifier_message: Fp4SmallM31) -> Vec<Fp4SmallM31> {
    // will use these in the loop
    let verifier_challenge_vector: Simd<u32, 4> =
        Simd::from_array(unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_message) });

    // generate out
    let out: Vec<Fp4SmallM31> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let a = src.get(2 * i).unwrap();
            let b = src.get((2 * i) + 1).unwrap();

            // (b - a)
            let b_minus_a = b - a;
            let b_minus_a_raw = unsafe { mem::transmute::<SmallM31, u32>(b_minus_a) };

            // verifier_message * (b - a)
            let tmp = mul_mod_m31_u32x4::<2_147_483_647>(
                verifier_challenge_vector,
                Simd::splat(b_minus_a_raw),
            );
            let mut raw = *tmp.as_array();

            // a + verifier_message * (b - a)
            let a_raw = unsafe { mem::transmute::<SmallM31, u32>(*a) };
            raw[0] = add_mod_val::<2_147_483_647>(raw[0], a_raw);

            unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(raw) }
        })
        .collect();
    out
}

#[cfg(test)]
mod tests {
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;

    use crate::multilinear::pairwise;
    use crate::tests::{Fp4SmallM31, SmallM31};
    use crate::wip::m31::reduce_bf::reduce_bf;

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let mut src: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
        let src_copy: Vec<SmallM31> = src.clone();
        let challenge_bf = SmallM31::from(7);
        let challenge_ef = Fp4SmallM31::from_base_prime_field(challenge_bf);

        // run function
        pairwise::reduce_evaluations(&mut src, challenge_bf);
        let expected_ef: Vec<Fp4SmallM31> = src
            .into_iter()
            .map(Fp4SmallM31::from_base_prime_field)
            .collect();
        let received_ef = reduce_bf(&src_copy, challenge_ef);

        assert_eq!(expected_ef, received_ef);
    }
}
