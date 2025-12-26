use ark_std::{cfg_into_iter, mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    experimental::goldilocks::arithmetic::{add::add, mul::mul},
    tests::Fp2SmallGoldilocks,
};

#[inline(always)]
pub fn mul_fp2_smallgoldilocks(scalar: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    let [a0, a1] = scalar;
    let [b0, b1] = b;

    // Standard schoolbook multiplication for extensions:
    // c0 = a0*b0 + non_residue * a1*b1
    // c1 = a0*b1 + a1*b0
    
    let a0b0 = mul(a0, b0);
    let a1b1 = mul(a1, b1);
    let a0b1 = mul(a0, b1);
    let a1b0 = mul(a1, b0);

    // Goldilocks non-residue is 7
    let non_residue_times_a1b1 = mul(a1b1, 7);

    let c0 = add(a0b0, non_residue_times_a1b1);
    let c1 = add(a0b1, a1b0);

    [c0, c1]
}

pub fn reduce_ef(src: &mut Vec<Fp2SmallGoldilocks>, verifier_message: Fp2SmallGoldilocks) {
    // will use these in the loop
    let verifier_message_raw = unsafe { mem::transmute::<Fp2SmallGoldilocks, [u64; 2]>(verifier_message) };

    // generate out
    let out: Vec<Fp2SmallGoldilocks> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let a = src.get(2 * i).unwrap();
            let b = src.get((2 * i) + 1).unwrap();

            // (b - a)
            let b_minus_a = b - a;
            let b_minus_a_raw = unsafe { mem::transmute::<Fp2SmallGoldilocks, [u64; 2]>(b_minus_a) };

            // verifier_message * (b - a)
            let tmp0 = mul_fp2_smallgoldilocks(verifier_message_raw, b_minus_a_raw);

            // a + verifier_message * (b - a)
            let mut tmp1 = unsafe { mem::transmute::<[u64; 2], Fp2SmallGoldilocks>(tmp0) };
            tmp1 += a;

            tmp1
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
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let mut expected_ef: Vec<Fp2SmallGoldilocks> =
            (0..LEN).map(|_| Fp2SmallGoldilocks::rand(&mut rng)).collect();
        let mut received_ef: Vec<Fp2SmallGoldilocks> = expected_ef.clone();
        let challenge_ef = Fp2SmallGoldilocks::from(7);

        // run function
        pairwise::reduce_evaluations(&mut expected_ef, challenge_ef);
        reduce_ef(&mut received_ef, challenge_ef);

        assert_eq!(expected_ef, received_ef);
    }
}