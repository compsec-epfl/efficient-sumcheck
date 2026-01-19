use ark_std::{cfg_into_iter, mem, simd::Simd};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use super::arithmetic::{add::add, mul::mul_v, sub::sub};
use crate::tests::{Fp2SmallGoldilocks, SmallGoldilocks};
use core::slice::from_raw_parts;
use rayon::prelude::*;

// computes (1 - challenge) * a + challenge * b -> a + challenge * ( b - a )
#[inline(always)]
fn single_compress(src: &[u64], challenge: &Simd<u64, 2>) -> Fp2SmallGoldilocks {
    let &[a, b, ..] = src else { unreachable!("src must have at least 2 elements") };
    
    let diff = Simd::splat(sub(b, a));
    // challenge * ( b - a )
    let multi = mul_v(&diff, challenge);

    let mut result = *multi.as_array();
    // a + challenge * ( b - a )
    result[0] = add(result[0], a);
    unsafe { mem::transmute(result) }
}

pub fn reduce_bf(src: &[SmallGoldilocks], verifier_message: Fp2SmallGoldilocks) -> Vec<Fp2SmallGoldilocks> {
    let challenge: Simd<u64, 2> = unsafe { mem::transmute(verifier_message) };
    // let src_scalars: &[u64] = unsafe {
    //     from_raw_parts(src.as_ptr() as *const u64, src.len()) 
    // };

    let src_scalars: &[u64] =
        unsafe { from_raw_parts(src.as_ptr() as *const _, src.len()) };
    
    // computes (1 - challenge) * a + challenge * b -> a + challenge * ( b - a ) for each pair
    cfg_into_iter!(src_scalars.par_chunks_exact(2))
        .map(|pair| single_compress(pair, &challenge))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Field};
    use crate::multilinear::pairwise::reduce_evaluations;

    #[test]
    #[should_panic(expected = "src must have at least 2 elements")]
    fn test_single_compress_panic() {
        let src = vec![10]; // Only 1 element
        let challenge = Simd::from_array([1, 1]);
        single_compress(&src, &challenge);
    }

    #[test]
    fn test_single_compress_extension() {
        // Testing with an extension challenge [alpha_0, alpha_1]
        // a = 10, b = 20, challenge = [2, 3]
        // Expected: [10 + 2*(20-10), 0 + 3*(20-10)] = [30, 30]
        let src = vec![10, 20];
        let challenge = Simd::from_array([2, 3]);
        
        let result = single_compress(&src, &challenge);
        let raw_result: [u64; 2] = unsafe { std::mem::transmute(result) };
        
        assert_eq!(raw_result, [30, 30]);
    }

    #[test]
    fn test_single_compress_identity() {
        let src = vec![100, 200];
        
        // Case 1: Challenge is 0 -> Result should be a ([100, 0])
        let challenge_zero = Simd::from_array([0, 0]);
        let res_zero = single_compress(&src, &challenge_zero);
        assert_eq!(unsafe { std::mem::transmute::<_, [u64; 2]>(res_zero) }, [100, 0]);

        // Case 2: Challenge is 1 -> Result should be b ([200, 0])
        // Calculation: 100 + 1 * (200 - 100) = 200
        let challenge_one = Simd::from_array([1, 0]);
        let res_one = single_compress(&src, &challenge_one);
        assert_eq!(unsafe { std::mem::transmute::<_, [u64; 2]>(res_one) }, [200, 0]);
    }

    #[test]
    fn test_reduce_bf_logic_zero_challenge() {
        // 1. Setup Source: [a, b, c, d]
        let src = vec![
            SmallGoldilocks::from(1), 
            SmallGoldilocks::from(2),
            SmallGoldilocks::from(3), 
            SmallGoldilocks::from(4)
        ];

        // 2. Setup Challenge: alpha = [0, 0] (The Zero Element)
        let challenge_raw = [0u64, 0u64];
        let verifier_message: Fp2SmallGoldilocks = unsafe { std::mem::transmute(challenge_raw) };

        // 3. Execute Reduction
        let result = reduce_bf(&src, verifier_message);

        // 4. Verify Dimensions
        assert_eq!(result.len(), 2);

        // 5. Verify Calculations
        // Formula: a + 0 * (b - a) = a
        // Pair 1 (1, 2) with alpha=0 -> [1, 0]
        // Pair 2 (3, 4) with alpha=0 -> [3, 0]
        let res_raw: Vec<[u64; 2]> = result.into_iter()
            .map(|val| unsafe { std::mem::transmute(val) })
            .collect();

        // Updated Assertions for Zero Challenge
        assert_eq!(res_raw[0], [1, 0]);
        assert_eq!(res_raw[1], [3, 0]);
    }

    #[test]
    fn test_reduce_bf_logic() {
        // 1. Setup Source: 4 Goldilocks elements [a, b, c, d]
        // These will be processed as pairs: (a, b) and (c, d)
        let src = vec![
            SmallGoldilocks::from(10), 
            SmallGoldilocks::from(20),
            SmallGoldilocks::from(30), 
            SmallGoldilocks::from(40)
        ];

        // 2. Setup Challenge: alpha = [2, 3]
        // This is an extension field element Fp2
        let challenge_raw = [2u64, 3u64];
        let verifier_message: Fp2SmallGoldilocks = unsafe { std::mem::transmute(challenge_raw) };

        // 3. Execute Reduction
        let result = reduce_bf(&src, verifier_message);

        // 4. Verify Dimensions
        // Input was 4, output should be 4/2 = 2
        assert_eq!(result.len(), src.len() >> 1);

        // 5. Verify Calculations
        // Pair 1: 10 + [2, 3] * (20 - 10) = [10 + 20, 0 + 30] = [30, 30]
        // Pair 2: 30 + [2, 3] * (40 - 30) = [30 + 20, 0 + 30] = [50, 30]
        let res_raw: Vec<[u64; 2]> = result.into_iter()
            .map(|val| unsafe { std::mem::transmute(val) })
            .collect();

        assert_eq!(res_raw[0], [30, 30]);
        assert_eq!(res_raw[1], [50, 30]);
    }


    #[test]
    fn single() {
        let pairwise_reduce_elems: [SmallGoldilocks; 2] = [SmallGoldilocks::from(100u64), SmallGoldilocks::from(250u64)];

        let mut reference_elems = pairwise_reduce_elems.to_vec();
        let reduce_elems_input = pairwise_reduce_elems;

        let challenge_val = 7u64;
        let challenge_bf = SmallGoldilocks::from(challenge_val);
        let challenge_ef = Fp2SmallGoldilocks::from_base_prime_field(challenge_bf);

        reduce_evaluations(&mut reference_elems, challenge_bf);
        let expected_ef: Vec<Fp2SmallGoldilocks> = reference_elems
            .into_iter()
            .map(Fp2SmallGoldilocks::from_base_prime_field)
            .collect();

        let received_ef = reduce_bf(&reduce_elems_input, challenge_ef);
        assert_eq!(received_ef, expected_ef);
    }

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let mut pairwise_reduce_elems: Vec<SmallGoldilocks> =
            (0..LEN).map(|_| SmallGoldilocks::rand(&mut rng)).collect();
        let reduce_elems: Vec<SmallGoldilocks> = pairwise_reduce_elems.clone();

        let challenge_bf = SmallGoldilocks::from(7);
        let challenge_ef = Fp2SmallGoldilocks::from_base_prime_field(challenge_bf);

        // run function
        reduce_evaluations(&mut pairwise_reduce_elems, challenge_bf);
        let expected_ef: Vec<Fp2SmallGoldilocks> = pairwise_reduce_elems
            .into_iter()
            .map(Fp2SmallGoldilocks::from_base_prime_field)
            .collect();

        let received_ef = reduce_bf(&reduce_elems, challenge_ef);
        assert_eq!(expected_ef, received_ef);
    }
}
