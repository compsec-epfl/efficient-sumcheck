use ark_std::{cfg_into_iter, mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    tests::{Fp4SmallM31, SmallM31},
    wip::m31::{arithmetic::mul::mul_v, evaluate_bf::add_mod_val},
};

pub fn reduce_bf(src: &[SmallM31], verifier_message: Fp4SmallM31) -> Vec<Fp4SmallM31> {
    // will use these in the loop
    let verifier_challenge_vector: Simd<u32, 4> =
        Simd::from_array(unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_message) });
    let modulus: Simd<u64, 4> = Simd::splat(2_147_483_647);

    // generate out
    let out: Vec<Fp4SmallM31> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let a = src.get(2 * i).unwrap();
            let b = src.get((2 * i) + 1).unwrap();

            // (b - a)
            let b_minus_a = b - a;
            let b_minus_a_raw = unsafe { mem::transmute::<SmallM31, u32>(b_minus_a) };

            // verifier_message * (b - a)
            let mut tmp = Simd::splat(b_minus_a_raw);
            tmp = mul_v(&tmp, &verifier_challenge_vector, &modulus);
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
