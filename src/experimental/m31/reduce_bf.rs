use ark_std::{
    cfg_into_iter, mem,
    simd::Simd,
};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    tests::{Fp4SmallM31, SmallM31},
    experimental::m31::arithmetic::{
        add::{add, add_v},
        mul::mul_v,
        sub::{sub, sub_v},
    },
};

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
    let verifier_challenge_vector: Simd<u32, 4> =
        Simd::from_array(unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_message) });
    let src_raw: &[u32] =
        unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u32, src.len()) };

    let out: Vec<Fp4SmallM31> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let start = 2 * i;
            let end = start + 2;
            single_compress(&src_raw[start..end], &verifier_challenge_vector)
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
    use crate::experimental::m31::reduce_bf::{reduce_bf};

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

// // this works but it's massively more slow
// #[inline(always)]
// fn special_compress<const LANES: usize, const LANES_DIV_TWO: usize, const LANES_DIV_FOUR: usize>(
//     src: &[u32],
//     challenge: &Simd<u32, LANES>,
// ) -> Vec<Fp4SmallM31>
// where
//     LaneCount<LANES>: SupportedLaneCount,
//     LaneCount<LANES_DIV_FOUR>: SupportedLaneCount,
// {
//     // Invariants we rely on (same as your asserts):
//     debug_assert_eq!(LANES_DIV_TWO, LANES / 2);
//     debug_assert_eq!(LANES_DIV_FOUR, LANES / 4);
//     debug_assert!(src.len() % LANES_DIV_TWO == 0);

//     // Each pair (a,b) -> 1 Fp4 -> 4 u32s.
//     let num_pairs = src.len() / 2;
//     let mut out_u32 = Vec::<u32>::with_capacity(num_pairs * 4);

//     for base in (0..src.len()).step_by(LANES_DIV_TWO) {
//         // ---- gather a and b into fixed-size arrays ----
//         let mut a_arr = [0u32; LANES_DIV_FOUR];
//         let mut b_arr = [0u32; LANES_DIV_FOUR];

//         for lane in 0..LANES_DIV_FOUR {
//             let idx_a = base + 2 * lane;
//             let idx_b = base + 2 * lane + 1;
//             a_arr[lane] = src[idx_a];
//             b_arr[lane] = src[idx_b];
//         }

//         let a_simd = Simd::<u32, LANES_DIV_FOUR>::from_array(a_arr);
//         let b_simd = Simd::<u32, LANES_DIV_FOUR>::from_array(b_arr);

//         // ---- (b - a) per lane ----
//         let b_minus_a = sub_v(&b_simd, &a_simd);
//         let b_minus_a_raw = b_minus_a.to_array(); // [u32; LANES_DIV_FOUR]

//         // ---- repeated: repeat_n(x, 4) on the stack ----
//         let mut repeated = [0u32; LANES];
//         for (lane, &x) in b_minus_a_raw.iter().enumerate() {
//             let base_lane = 4 * lane;
//             repeated[base_lane] = x;
//             repeated[base_lane + 1] = x;
//             repeated[base_lane + 2] = x;
//             repeated[base_lane + 3] = x;
//         }

//         let mut res = mul_v(challenge, &Simd::<u32, LANES>::from_array(repeated));

//         // ---- a_expanded: [a, 0, 0, 0] per lane on the stack ----
//         let mut a_expanded = [0u32; LANES];
//         for (lane, &x) in a_arr.iter().enumerate() {
//             let base_lane = 4 * lane;
//             a_expanded[base_lane] = x;
//             // others remain zero
//         }

//         res = add_v(&res, &Simd::<u32, LANES>::from_array(a_expanded));

//         out_u32.extend_from_slice(&res.to_array());
//     }

//     // Convert flat u32 limbs -> Vec<Fp4SmallM31> by chunking into [u32; 4]
//     out_u32
//         .chunks_exact(4)
//         .map(|w| unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>([w[0], w[1], w[2], w[3]]) })
//         .collect()
// }