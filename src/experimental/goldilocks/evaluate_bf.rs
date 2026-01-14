use std::simd::{LaneCount, SupportedLaneCount};

use ark_std::{mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, prelude::ParallelSlice};

use crate::{
    experimental::goldilocks::arithmetic::add::{add, add_v},
    tests::SmallGoldilocks,
};

#[inline(always)]
fn sum_v<const LANES: usize>(src: &[u64]) -> (u64, u64)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut acc0 = Simd::<u64, LANES>::splat(0);
    let mut acc1 = Simd::<u64, LANES>::splat(0);
    let mut acc2 = Simd::<u64, LANES>::splat(0);
    let mut acc3 = Simd::<u64, LANES>::splat(0);

    let chunk_size = 4 * LANES;
    for i in (0..src.len()).step_by(chunk_size) {
        acc0 = add_v(&acc0, &Simd::<u64, LANES>::from_slice(&src[i..i + LANES]));

        acc1 = add_v(
            &acc1,
            &Simd::<u64, LANES>::from_slice(&src[i + LANES..i + 2 * LANES]),
        );

        acc2 = add_v(
            &acc2,
            &Simd::<u64, LANES>::from_slice(&src[i + 2 * LANES..i + 3 * LANES]),
        );

        acc3 = add_v(
            &acc3,
            &Simd::<u64, LANES>::from_slice(&src[i + 3 * LANES..i + 4 * LANES]),
        );
    }

    #[inline(always)]
    fn sum_indexwise<const LANES: usize>(accs: &[[u64; LANES]]) -> (u64, u64) {
        let mut even_sum = 0u64;
        let mut odd_sum = 0u64;
        for i in (0..LANES).step_by(2) {
            for acc in accs {
                even_sum = add(even_sum, acc[i]);
                odd_sum = add(odd_sum, acc[i + 1]);
            }
        }
        (even_sum, odd_sum)
    }

    sum_indexwise(&[
        acc0.to_array(),
        acc1.to_array(),
        acc2.to_array(),
        acc3.to_array(),
    ])
}

pub fn evaluate_bf<const MODULUS: u64>(
    src: &[SmallGoldilocks],
) -> (SmallGoldilocks, SmallGoldilocks) {
    const CHUNK_SIZE: usize = 32_768;
    let sums = src
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let chunk_raw: &[u64] =
                unsafe { core::slice::from_raw_parts(chunk.as_ptr() as *const u64, chunk.len()) };
            sum_v::<16>(chunk_raw)
        })
        .reduce(|| (0, 0), |(e1, o1), (e2, o2)| (add(e1, e2), add(o1, o2)));

    let (sum_0_u64, sum_1_u64) = sums;
    let sum_0: SmallGoldilocks = unsafe { mem::transmute::<u64, SmallGoldilocks>(sum_0_u64) };
    let sum_1: SmallGoldilocks = unsafe { mem::transmute::<u64, SmallGoldilocks>(sum_1_u64) };
    (sum_0, sum_1)
}

#[cfg(test)]
mod tests {
    use super::super::MODULUS;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use super::evaluate_bf;
    use crate::multilinear::pairwise;
    use crate::tests::SmallGoldilocks;

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 4 * 16;
        let mut rng = test_rng();

        // random elements
        let src: Vec<SmallGoldilocks> = (0..LEN).map(|_| SmallGoldilocks::rand(&mut rng)).collect();

        // run function
        let expected = pairwise::evaluate(&src);

        let received = evaluate_bf::<MODULUS>(&src);

        assert_eq!(expected, received);
    }
}
