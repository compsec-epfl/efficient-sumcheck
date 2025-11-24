use std::simd::{LaneCount, SupportedLaneCount};

use ark_std::{mem, simd::Simd, slice};
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, prelude::ParallelSlice};

use crate::{experimental::m31::arithmetic::add::add_v, tests::Fp4SmallM31};

#[inline(always)]
fn sum_v<const LANES: usize>(src: &[u32]) -> ([u32; 4], [u32; 4])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut acc0 = Simd::<u32, LANES>::splat(0);
    let mut acc1 = Simd::<u32, LANES>::splat(0);
    let mut acc2 = Simd::<u32, LANES>::splat(0);
    let mut acc3 = Simd::<u32, LANES>::splat(0);

    let chunk_size = 4 * LANES; // NOTE: unroll by 4 bc why not
    let fast_path_end = src.len() - (src.len() % chunk_size);
    for i in (0..fast_path_end).step_by(chunk_size) {
        acc0 = add_v(&acc0, &Simd::<u32, LANES>::from_slice(&src[i..i + LANES]));
        acc1 = add_v(
            &acc1,
            &Simd::<u32, LANES>::from_slice(&src[i + LANES..i + 2 * LANES]),
        );
        acc2 = add_v(
            &acc2,
            &Simd::<u32, LANES>::from_slice(&src[i + 2 * LANES..i + 3 * LANES]),
        );
        acc3 = add_v(
            &acc3,
            &Simd::<u32, LANES>::from_slice(&src[i + 3 * LANES..i + 4 * LANES]),
        );
    }

    #[inline(always)]
    fn sum_fp4wise(accs: &[&[u32]]) -> ([u32; 4], [u32; 4]) {
        let mut even_sum = Simd::<u32, 4>::splat(0);
        let mut odd_sum = Simd::<u32, 4>::splat(0);
        let mut is_even = true;
        for acc in accs {
            for j in (0..acc.len()).step_by(4) {
                if is_even {
                    even_sum = add_v(&even_sum, &Simd::from_slice(&acc[j..j + 4]));
                } else {
                    odd_sum = add_v(&odd_sum, &Simd::from_slice(&acc[j..j + 4]));
                }
                is_even = !is_even;
            }
        }
        (even_sum.to_array(), odd_sum.to_array())
    }

    sum_fp4wise(&[
        &acc0.to_array(),
        &acc1.to_array(),
        &acc2.to_array(),
        &acc3.to_array(),
        &src[fast_path_end..],
    ])
}

pub fn evaluate_ef<const MODULUS: u32>(src: &[Fp4SmallM31]) -> (Fp4SmallM31, Fp4SmallM31) {
    const CHUNK_SIZE: usize = 32_768;
    let sums = src
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let chunk_raw: &[u32] =
                unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u32, chunk.len() * 4) };
            sum_v::<32>(chunk_raw)
        })
        .reduce(
            || ([0u32; 4], [0u32; 4]),
            |(e1, o1), (e2, o2)| {
                (
                    add_v(&Simd::from_array(e1), &Simd::from_array(e2)).to_array(),
                    add_v(&Simd::from_array(o1), &Simd::from_array(o2)).to_array(),
                )
            },
        );

    let (sum0_raw, sum1_raw) = sums;
    let sum0: Fp4SmallM31 = unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(sum0_raw) };
    let sum1: Fp4SmallM31 = unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(sum1_raw) };
    (sum0, sum1)
}

#[cfg(test)]
mod tests {
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;

    use crate::experimental::m31::evaluate_ef::evaluate_ef;
    use crate::multilinear::pairwise;
    use crate::tests::{Fp4SmallM31, SmallM31};

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let src_bf: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();
        let src_ef: Vec<Fp4SmallM31> = src_bf
            .clone()
            .into_iter()
            .map(Fp4SmallM31::from_base_prime_field)
            .collect();

        // run function
        let expected_bf = pairwise::evaluate(&src_bf);
        let expected_ef = (
            Fp4SmallM31::from_base_prime_field(expected_bf.0),
            Fp4SmallM31::from_base_prime_field(expected_bf.1),
        );
        let received_ef = evaluate_ef::<2_147_483_647>(&src_ef);

        assert_eq!(expected_ef, received_ef);
    }
}
