use ark_std::{cfg_chunks, cfg_into_iter, mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    experimental::goldilocks::arithmetic::{add::add, mul::mul_v, sub::sub},
    tests::{Fp2SmallGoldilocks, SmallGoldilocks},
};

#[inline(always)]
fn single_compress(src: &[u64], challenge: &Simd<u64, 2>) -> Fp2SmallGoldilocks {
    let a = src.first().unwrap();
    let b = src.get(1).unwrap();

    let b_minus_a = sub(*b, *a);

    let mut tmp = Simd::splat(b_minus_a);
    tmp = mul_v(&tmp, challenge);
    let mut raw = *tmp.as_array();

    raw[0] = add(raw[0], *a);
    unsafe { mem::transmute::<[u64; 2], Fp2SmallGoldilocks>(raw) }
}

pub fn reduce_bf(
    src: &[SmallGoldilocks],
    verifier_message: Fp2SmallGoldilocks,
) -> Vec<Fp2SmallGoldilocks> {
    let verifier_challenge_raw =
        unsafe { mem::transmute::<Fp2SmallGoldilocks, [u64; 2]>(verifier_message) };

    let src_raw: &[u64] =
        unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u64, src.len()) };

    let verifier_challenge_vector: Simd<u64, 2> = Simd::from_array(verifier_challenge_raw);
    let out: Vec<Fp2SmallGoldilocks> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let start = 2 * i;
            let end = start + 2;
            single_compress(&src_raw[start..end], &verifier_challenge_vector)
        })
        .collect();
    out
}

// has error
#[cfg(test)]
mod tests {
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;

    use super::reduce_bf;
    use crate::multilinear::pairwise::reduce_evaluations;
    use crate::tests::{Fp2SmallGoldilocks, SmallGoldilocks};

    #[test]
    fn sanity_static_values() {
        let pairwise_reduce_elems: [SmallGoldilocks; 2] =
            [SmallGoldilocks::from(100u64), SmallGoldilocks::from(250u64)];

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
