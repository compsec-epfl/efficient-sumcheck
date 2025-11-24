use ark_std::{cfg_chunks, cfg_into_iter, mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    prelude::ParallelSlice,
};

use crate::{
    experimental::m31::arithmetic::{
        add::add,
        mul::mul_v,
        sub::{sub, sub_v},
    },
    tests::{Fp4SmallM31, SmallM31},
};

#[inline(always)]
fn double_compress(src: &[u32], challenge: &Simd<u32, 8>) -> [Fp4SmallM31; 2] {
    let b_minus_a: Simd<u32, 4> = Simd::splat(sub(src[1], src[0]));
    let d_minus_c: Simd<u32, 4> = Simd::splat(sub(src[3], src[2]));
    let combined = Simd::<u32, 8>::from_array([
        b_minus_a[0],
        b_minus_a[1],
        b_minus_a[2],
        b_minus_a[3],
        d_minus_c[0],
        d_minus_c[1],
        d_minus_c[2],
        d_minus_c[3],
    ]);

    let res = mul_v(challenge, &combined);
    let mut raw = *res.as_array();

    raw[0] = add(raw[0], src[0]);
    raw[4] = add(raw[4], src[2]);

    unsafe { mem::transmute::<[u32; 8], [Fp4SmallM31; 2]>(raw) }
}

#[inline(always)]
fn single_compress(src: &[u32], challenge: &Simd<u32, 4>) -> Fp4SmallM31 {
    let a = src.first().unwrap();
    let b = src.get(1).unwrap();

    let b_minus_a = sub(*b, *a);

    let mut tmp = Simd::splat(b_minus_a);
    tmp = mul_v(&tmp, challenge);
    let mut raw = *tmp.as_array();

    raw[0] = add(raw[0], *a);

    unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(raw) }
}

pub fn reduce_bf(src: &[SmallM31], verifier_message: Fp4SmallM31) -> Vec<Fp4SmallM31> {
    let verifier_challenge_raw =
        unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_message) };
    let src_raw: &[u32] =
        unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u32, src.len()) };

    let verifier_challenge_vector: Simd<u32, 4> = Simd::from_array(verifier_challenge_raw);
    let out: Vec<Fp4SmallM31> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let start = 2 * i;
            let end = start + 2;
            single_compress(&src_raw[start..end], &verifier_challenge_vector)
        })
        .collect();

    // This works but it's way slower
    // let verifier_challenge_vector: Simd<u32, 8> = Simd::from_array([
    //     verifier_challenge_raw[0],
    //     verifier_challenge_raw[1],
    //     verifier_challenge_raw[2],
    //     verifier_challenge_raw[3],
    //     verifier_challenge_raw[0],
    //     verifier_challenge_raw[1],
    //     verifier_challenge_raw[2],
    //     verifier_challenge_raw[3],
    // ]);
    // let out: Vec<Fp4SmallM31> = cfg_into_iter!(0..src_raw.len() / 4)
    //     .flat_map(|i| {
    //         let start = 4 * i;
    //         let end = start + 4;
    //         double_compress(&src_raw[start..end], &verifier_challenge_vector)
    //     })
    //     .collect();
    out
}

#[cfg(test)]
mod tests {
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;

    use crate::experimental::m31::reduce_bf::reduce_bf;
    use crate::multilinear::pairwise;
    use crate::tests::{Fp4SmallM31, SmallM31};

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

    // #[test]
    // fn compress() {
    //     // setup
    //     const LEN: usize = 1 << 4;
    //     let mut rng = test_rng();

    //     // random elements
    //     let src: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
    //     let challenge = Fp4SmallM31::rand(&mut rng);
    //     let challenge_raw: [u32; 4] = unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(challenge) };
    //     let doubled: [u32; 8] = [
    //         challenge_raw[0],
    //         challenge_raw[1],
    //         challenge_raw[2],
    //         challenge_raw[3],
    //         challenge_raw[0],
    //         challenge_raw[1],
    //         challenge_raw[2],
    //         challenge_raw[3],
    //     ];

    //     let expected: Vec<Fp4SmallM31> = (0..src.len() / 2)
    //         .map(|i| {
    //             let input: Vec<SmallM31> = src[2 * i..=(2 * i) + 1].to_vec();
    //             let compressed = single_compress(
    //                 unsafe {
    //                     core::slice::from_raw_parts(input.as_ptr() as *const u32, input.len())
    //                 },
    //                 &Simd::from_array(challenge_raw),
    //             );
    //             compressed
    //         })
    //         .collect();

    //     let received: Vec<Fp4SmallM31> = (0..src.len())
    //         .step_by(8)
    //         .flat_map(|i| {
    //             let input: Vec<SmallM31> = src[i..i + 8].to_vec();
    //             let compressed: Vec<Fp4SmallM31> = special_compress::<8, 4, 2>(
    //                 unsafe {
    //                     core::slice::from_raw_parts(input.as_ptr() as *const u32, input.len())
    //                 },
    //                 &Simd::from_array(doubled),
    //             );
    //             compressed
    //         })
    //         .collect();

    //     assert_eq!(expected, received);
    // }
}
