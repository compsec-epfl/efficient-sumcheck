use ark_std::{cfg_into_iter, mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{tests::Fp4SmallM31, wip::m31::{evaluate_bf::{add_mod_val, mul_mod_val}}};

#[inline(always)]
fn mul_fp4_smallm31(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;

    // base field ops: plug your own
    let add = |x, y| add_mod_val::<2_147_483_647>(x, y);
    let mul = |x, y| mul_mod_val::<2_147_483_647>(x, y);

    // A0*B0
    let t0 = mul(a0, b0);
    let t1 = mul(a1, b1);
    let a0b0_0 = add(t0, mul(t1, 3));
    let a0b0_1 = add(mul(a0, b1), mul(a1, b0));

    // A1*B1
    let u0 = mul(a2, b2);
    let u1 = mul(a3, b3);
    let a1b1_0 = add(u0, mul(u1, 3));
    let a1b1_1 = add(mul(a2, b3), mul(a3, b2));

    // β * A1*B1, β = (3, 0)
    let beta_a1b1_0 = mul(a1b1_0, 3);
    let beta_a1b1_1 = mul(a1b1_1, 3);

    // C0
    let c0_0 = add(a0b0_0, beta_a1b1_0);
    let c0_1 = add(a0b0_1, beta_a1b1_1);

    // A0*B1
    let v0 = mul(a0, b2);
    let v1 = mul(a1, b3);
    let a0b1_0 = add(v0, mul(v1, 3));
    let a0b1_1 = add(mul(a0, b3), mul(a1, b2));

    // A1*B0
    let w0 = mul(a2, b0);
    let w1 = mul(a3, b1);
    let a1b0_0 = add(w0, mul(w1, 3));
    let a1b0_1 = add(mul(a2, b1), mul(a3, b0));

    // C1
    let c1_0 = add(a0b1_0, a1b0_0);
    let c1_1 = add(a0b1_1, a1b0_1);

    [c0_0, c0_1, c1_0, c1_1]
}

pub fn reduce_ef(src: &mut Vec<Fp4SmallM31>, verifier_message: Fp4SmallM31) {
    // will use these in the loop
    let verifier_message_raw = unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_message) };
    let verifier_message_vector: Simd<u32, 4> = Simd::from_array(verifier_message_raw);

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
            assert_eq!(tmp0, (verifier_message_vector * Simd::from_array(b_minus_a_raw)).to_array());

            // a + verifier_message * (b - a)
            let a_raw = unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(*a) };
            let tmp1 = Simd::from_array(tmp0) + Simd::from_array(a_raw);

            unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(*tmp1.as_array()) }
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
