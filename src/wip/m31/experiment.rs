use core::simd::Simd;

use crate::wip::m31::arithmetic::{add::add_v, mul::mul_v};

#[inline(always)]
pub fn mul_fp4_smallm31_2(scalar: [u32; 4], b: [u32; 4], c: [u32; 4]) -> ([u32; 4], [u32; 4]) {
    let [a0, a1, a2, a3] = scalar;
    let [b0, b1, b2, b3] = b;
    let [c0, c1, c2, c3] = c;

    // A0*B0
    let t = mul_v(
        &Simd::from_array([a0, a1, a0, a1]),
        &Simd::from_array([b0, b1, c0, c1]),
        &Simd::splat(2_147_483_647),
    );
    let t_prime = mul_v(
        &Simd::from_array([a0, a1, a0, a1]),
        &Simd::from_array([b1, b0, c1, c0]),
        &Simd::splat(2_147_483_647),
    );
    let a0b0 = add_v(
        &Simd::from_array([t[0], t_prime[0], t[2], t_prime[2]]),
        &Simd::from_array([t[1], t_prime[1], t[3], t_prime[3]]),
        &Simd::splat(2_147_483_647),
    );

    // A1*B1
    let u = mul_v(
        &Simd::from_array([a2, a3, a2, a3]),
        &Simd::from_array([b2, b3, c2, c3]),
        &Simd::splat(2_147_483_647),
    );
    let u_prime = mul_v(
        &Simd::from_array([a2, a3, a2, a3]),
        &Simd::from_array([b3, b2, c3, c2]),
        &Simd::splat(2_147_483_647),
    );
    let a1b1 = add_v(
        &Simd::from_array([u[0], u_prime[0], u[2], u_prime[2]]),
        &Simd::from_array([u[1], u_prime[1], u[3], u_prime[3]]),
        &Simd::splat(2_147_483_647),
    );

    // C0
    let C0 = add_v(
        &Simd::from_array([a0b0[0], a0b0[1], a0b0[2], a0b0[3]]),
        &Simd::from_array([a1b1[0], a1b1[1], a1b1[2], a1b1[3]]),
        &Simd::splat(2_147_483_647),
    );

    // A0*B1
    let v = mul_v(
        &Simd::from_array([a0, a1, a0, a1]),
        &Simd::from_array([b2, b3, c2, c3]),
        &Simd::splat(2_147_483_647),
    );
    let v_prime = mul_v(
        &Simd::from_array([a0, a1, a0, a1]),
        &Simd::from_array([b3, b2, c3, c2]),
        &Simd::splat(2_147_483_647),
    );
    let a0b1 = add_v(
        &Simd::from_array([v[0], v_prime[0], v[2], v_prime[2]]),
        &Simd::from_array([v[1], v_prime[1], v[3], v_prime[3]]),
        &Simd::splat(2_147_483_647),
    );

    // A1*B0
    let w = mul_v(
        &Simd::from_array([a2, a3, a2, a3]),
        &Simd::from_array([b0, b1, c0, c1]),
        &Simd::splat(2_147_483_647),
    );
    let w_prime = mul_v(
        &Simd::from_array([a2, a3, a2, a3]),
        &Simd::from_array([b1, b0, c1, c0]),
        &Simd::splat(2_147_483_647),
    );
    let a1b0 = add_v(
        &Simd::from_array([w[0], w_prime[0], w[2], w_prime[2]]),
        &Simd::from_array([w[1], w_prime[1], w[3], w_prime[3]]),
        &Simd::splat(2_147_483_647),
    );

    // C1
    let C1 = add_v(
        &Simd::from_array([a0b1[0], a0b1[1], a0b1[2], a0b1[3]]),
        &Simd::from_array([a1b0[0], a1b0[1], a1b0[2], a1b0[3]]),
        &Simd::splat(2_147_483_647),
    );

    ([C0[0], C0[1], C1[0], C1[1]], [C0[2], C0[3], C1[2], C1[3]])
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use ark_std::{mem, test_rng};

    use crate::{tests::Fp4SmallM31, wip::m31::reduce_ef::mul_fp4_smallm31};

    use super::mul_fp4_smallm31_2;

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let src: Vec<Fp4SmallM31> = (0..LEN).map(|_| Fp4SmallM31::rand(&mut rng)).collect();
        let verifier_message = Fp4SmallM31::from(7);
        let verifier_message_raw =
            unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(verifier_message) };

        let expected: Vec<Fp4SmallM31> = src
            .iter()
            .map(|x| {
                let raw_x: [u32; 4] = unsafe { core::mem::transmute::<Fp4SmallM31, [u32; 4]>(*x) };

                let raw_res: [u32; 4] = mul_fp4_smallm31(verifier_message_raw, raw_x);

                unsafe { core::mem::transmute::<[u32; 4], Fp4SmallM31>(raw_res) }
            })
            .collect();

        let received: Vec<Fp4SmallM31> = src
            .chunks_exact(2) // &[Fp4SmallM31] of length 2 each
            .flat_map(|chunk| {
                // convert pair of Fp4SmallM31 → [[u32; 4]; 2]
                let raw_b = unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(chunk[0]) };
                let raw_c = unsafe { mem::transmute::<Fp4SmallM31, [u32; 4]>(chunk[1]) };

                // SIMD batch: N = 2, LANES = 4
                let (out_b, out_c) = mul_fp4_smallm31_2(verifier_message_raw, raw_b, raw_c);

                // return an array of 2 Fp4SmallM31, which flat_map will flatten
                [
                    unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(out_b) },
                    unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(out_c) },
                ]
            })
            .collect();

        assert_eq!(expected, received);
    }
}
