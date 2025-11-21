use ark_std::{cfg_into_iter, mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    tests::Fp4SmallM31,
    wip::m31::arithmetic::{add::add_v, mul::mul_v},
};

#[inline(always)]
pub fn mul_fp4_smallm31(scalar: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    let [a0, a1, a2, a3] = scalar;
    let [b0, b1, b2, b3] = b;

    // A0*B0
    let t = mul_v(&Simd::from_array([a0, a1]), &Simd::from_array([b0, b1]));
    let t_prime = mul_v(&Simd::from_array([a0, a1]), &Simd::from_array([b1, b0]));
    let a0b0 = add_v(
        &Simd::from_array([t[0], t_prime[0]]),
        &Simd::from_array([t[1], t_prime[1]]),
    );

    // A1*B1
    let u = mul_v(&Simd::from_array([a2, a3]), &Simd::from_array([b2, b3]));
    let u_prime = mul_v(&Simd::from_array([a2, a3]), &Simd::from_array([b3, b2]));
    let a1b1 = add_v(
        &Simd::from_array([u[0], u_prime[0]]),
        &Simd::from_array([u[1], u_prime[1]]),
    );

    // C0
    let c0 = add_v(
        &Simd::from_array([a0b0[0], a0b0[1]]),
        &Simd::from_array([a1b1[0], a1b1[1]]),
    );

    // A0*B1
    let v = mul_v(&Simd::from_array([a0, a1]), &Simd::from_array([b2, b3]));
    let v_prime = mul_v(&Simd::from_array([a0, a1]), &Simd::from_array([b3, b2]));
    let a0b1 = add_v(
        &Simd::from_array([v[0], v_prime[0]]),
        &Simd::from_array([v[1], v_prime[1]]),
    );

    // A1*B0
    let w = mul_v(&Simd::from_array([a2, a3]), &Simd::from_array([b0, b1]));
    let w_prime = mul_v(&Simd::from_array([a2, a3]), &Simd::from_array([b1, b0]));
    let a1b0 = add_v(
        &Simd::from_array([w[0], w_prime[0]]),
        &Simd::from_array([w[1], w_prime[1]]),
    );

    // C1
    let c1 = add_v(
        &Simd::from_array([a0b1[0], a0b1[1]]),
        &Simd::from_array([a1b0[0], a1b0[1]]),
    );

    [c0[0], c0[1], c1[0], c1[1]]
}

pub fn reduce_ef(src: &mut Vec<Fp4SmallM31>, verifier_message: Fp4SmallM31) {
    // will use these in the loop
    let verifier_message_raw = unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_message) };
    // let verifier_message_vector: Simd<u32, 4> = Simd::from_array(verifier_message_raw);

    // generate out
    let out: Vec<Fp4SmallM31> = cfg_into_iter!(0..src.len() / 2)
        .map(|i| {
            let a = src.get(2 * i).unwrap();
            let b = src.get((2 * i) + 1).unwrap();

            // (b - a)
            let b_minus_a = b - a;
            let b_minus_a_raw = unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(b_minus_a) };

            // verifier_message * (b - a)
            let tmp0 = mul_fp4_smallm31(verifier_message_raw, b_minus_a_raw);

            // a + verifier_message * (b - a)
            let mut tmp1 = unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(tmp0) };
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

    use crate::multilinear::pairwise;
    use crate::tests::Fp4SmallM31;
    use crate::wip::m31::reduce_ef::reduce_ef;

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let mut expected_ef: Vec<Fp4SmallM31> =
            (0..LEN).map(|_| Fp4SmallM31::rand(&mut rng)).collect();
        let mut received_ef: Vec<Fp4SmallM31> = expected_ef.clone();
        let challenge_ef = Fp4SmallM31::from(7);

        // run function
        pairwise::reduce_evaluations(&mut expected_ef, challenge_ef);
        reduce_ef(&mut received_ef, challenge_ef);

        assert_eq!(expected_ef, received_ef);
    }
}
