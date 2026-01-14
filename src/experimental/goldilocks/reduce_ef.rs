use ark_std::{cfg_into_iter, mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    experimental::goldilocks::arithmetic::{add::add_v, mul::mul_v, sub::sub_v},
    tests::Fp2SmallGoldilocks,
};

#[inline(always)]
pub fn mul_fp2_goldilocks(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    let [a0, a1] = a;
    let [b0, b1] = b;

    // 1. Calculate base products using SIMD
    let ab_prod = mul_v(
        &Simd::<u64, 2>::from_array([a0, a1]),
        &Simd::<u64, 2>::from_array([b0, b1]),
    );
    let ac = ab_prod[0];
    let bd = ab_prod[1];

    let cross_prod = mul_v(
        &Simd::<u64, 2>::from_array([a0, a1]),
        &Simd::<u64, 2>::from_array([b1, b0]),
    );
    let ad = cross_prod[0];
    let bc = cross_prod[1];

    // 2. Non-residue term: 3 * bd
    // Using a constant 3 here instead of 7
    let term_bd_3 = mul_v(&Simd::<u64, 1>::splat(bd), &Simd::<u64, 1>::splat(3))[0];

    // 3. Real part: ac + 3*bd
    let real = add_v(
        &Simd::<u64, 1>::splat(ac),
        &Simd::<u64, 1>::splat(term_bd_3),
    )[0];

    // 4. Imaginary part: ad + bc
    let imag = add_v(&Simd::<u64, 1>::splat(ad), &Simd::<u64, 1>::splat(bc))[0];

    [real, imag]
}

pub fn reduce_ef(src: &mut Vec<Fp2SmallGoldilocks>, verifier_message: Fp2SmallGoldilocks) {
    let verifier_message_raw =
        unsafe { mem::transmute::<Fp2SmallGoldilocks, [u64; 2]>(verifier_message) };

    let out: Vec<Fp2SmallGoldilocks> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let a = src.get(2 * i).unwrap();
            let b = src.get((2 * i) + 1).unwrap();

            // (b - a)
            let b_minus_a = b - a;
            let b_minus_a_raw =
                unsafe { mem::transmute::<Fp2SmallGoldilocks, [u64; 2]>(b_minus_a) };

            // verifier_message * (b - a)
            let result_raw = mul_fp2_goldilocks(verifier_message_raw, b_minus_a_raw);

            // a + verifier_message * (b - a)
            let mut result = unsafe { mem::transmute::<[u64; 2], Fp2SmallGoldilocks>(result_raw) };
            result += a;

            result
        })
        .collect();

    // write back into src
    src[..out.len()].copy_from_slice(&out);
    src.truncate(out.len());
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use crate::experimental::goldilocks::reduce_ef::reduce_ef;
    use crate::multilinear::pairwise;
    use crate::tests::Fp2SmallGoldilocks;

    #[test]
    fn static_goldilocks_fp2_reduction2() {
        use ark_ff::Field;
        // Use the proper constructor to handle Montgomery encoding
        let a = Fp2SmallGoldilocks::new(10u64.into(), 2u64.into());
        let b = Fp2SmallGoldilocks::new(15u64.into(), 5u64.into());
        let r = Fp2SmallGoldilocks::new(7u64.into(), 3u64.into());

        // Manual calculation expected result (also in Montgomery space)
        let expected = Fp2SmallGoldilocks::new(72u64.into(), 38u64.into());

        let mut expected_vec = vec![a, b];
        pairwise::reduce_evaluations(&mut expected_vec, r);

        let mut received_vec = vec![a, b];
        reduce_ef(&mut received_vec, r);

        assert_eq!(
            received_vec[0], expected_vec[0],
            "SIMD vs Reference mismatch"
        );
        assert_eq!(received_vec[0], expected, "SIMD vs Manual mismatch");
    }

    #[test]
    fn static_goldilocks_fp2_reduction() {
        // Define raw coordinates: [real, imag]
        // We use values that don't require complex modular reduction yet to isolate logic
        let a_raw = [10u64, 2u64];
        let b_raw = [15u64, 5u64];
        let r_raw = [7u64, 3u64]; // Challenge with an imaginary part to test non-residue

        // Expected result based on i^2 = 3: [72, 38]
        let expected_raw = [72u64, 38u64];

        // Convert to Fp2SmallGoldilocks
        let a: Fp2SmallGoldilocks = unsafe { std::mem::transmute(a_raw) };
        let b: Fp2SmallGoldilocks = unsafe { std::mem::transmute(b_raw) };
        let r: Fp2SmallGoldilocks = unsafe { std::mem::transmute(r_raw) };
        let expected: Fp2SmallGoldilocks = unsafe { std::mem::transmute(expected_raw) };
        println!("{}", expected);

        // Reference implementation
        let mut expected_vec = vec![a, b];
        pairwise::reduce_evaluations(&mut expected_vec, r);

        // Your SIMD/Optimized implementation
        let mut received_vec = vec![a, b];
        reduce_ef(&mut received_vec, r);

        // Final assertions
        // assert_eq!(expected_vec.len(), 1);
        // assert_eq!(received_vec.len(), 1);
        assert_eq!(
            received_vec[0], expected,
            "SIMD result does not match manual calculation! Check your i^2 non-residue logic."
        );
        // assert_eq!(
        //     received_vec[0],
        //     expected_vec[0],
        //     "SIMD result does not match pairwise reference!"
        // );
        //   left: QuadExtField { c0: 18446743760176939009, c1: 18446743906205827073 }
    }

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let mut expected_ef: Vec<Fp2SmallGoldilocks> = (0..LEN)
            .map(|_| Fp2SmallGoldilocks::rand(&mut rng))
            .collect();

        let mut received_ef: Vec<Fp2SmallGoldilocks> = expected_ef.clone();
        let challenge_ef = Fp2SmallGoldilocks::from(7);

        // run function
        pairwise::reduce_evaluations(&mut expected_ef, challenge_ef);
        reduce_ef(&mut received_ef, challenge_ef);

        assert_eq!(expected_ef, received_ef);
    }
}
