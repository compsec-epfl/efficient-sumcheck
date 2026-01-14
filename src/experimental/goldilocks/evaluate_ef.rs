use std::simd::{LaneCount, SupportedLaneCount};

use crate::{experimental::goldilocks::arithmetic::add::add_v, tests::Fp2SmallGoldilocks};
use ark_std::{mem, simd::Simd, slice};
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, prelude::ParallelSlice};

#[inline(always)]
fn sum_v<const LANES: usize>(src: &[u64]) -> ([u64; 2], [u64; 2])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut acc0 = Simd::<u64, LANES>::splat(0);
    let mut acc1 = Simd::<u64, LANES>::splat(0);

    // We process 2 * LANES to keep the accumulators independent for ILP
    let chunk_size = 2 * LANES;
    let fast_path_end = src.len() - (src.len() % chunk_size);

    for i in (0..fast_path_end).step_by(chunk_size) {
        // add_v here is your Goldilocks modular addition
        acc0 = add_v(&acc0, &Simd::<u64, LANES>::from_slice(&src[i..i + LANES]));
        acc1 = add_v(
            &acc1,
            &Simd::<u64, LANES>::from_slice(&src[i + LANES..i + 2 * LANES]),
        );
    }

    // This handles the "Horizontal" reduction to separate Even/Odd FP2 elements
    #[inline(always)]
    fn sum_fp2wise(accs: &[&[u64]]) -> ([u64; 2], [u64; 2]) {
        let mut even_sum = Simd::<u64, 2>::splat(0); // [real, imag]
        let mut odd_sum = Simd::<u64, 2>::splat(0);
        let mut is_even = true;

        for acc in accs {
            for j in (0..acc.len()).step_by(2) {
                let val = Simd::from_slice(&acc[j..j + 2]);
                if is_even {
                    even_sum = add_v(&even_sum, &val);
                } else {
                    odd_sum = add_v(&odd_sum, &val);
                }
                is_even = !is_even;
            }
        }
        (even_sum.to_array(), odd_sum.to_array())
    }

    sum_fp2wise(&[&acc0.to_array(), &acc1.to_array(), &src[fast_path_end..]])
}

pub fn evaluate_ef<const MODULUS: u64>(
    src: &[Fp2SmallGoldilocks],
) -> (Fp2SmallGoldilocks, Fp2SmallGoldilocks) {
    const CHUNK_SIZE: usize = 16_384;
    let sums = src
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            // Treat the slice of FP2 elements as a raw slice of u64
            let chunk_raw: &[u64] =
                unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u64, chunk.len() * 2) };
            sum_v::<16>(chunk_raw) // Using 16 lanes for AVX-512 (1024-bit) or 8 for AVX2
        })
        .reduce(
            || ([0u64; 2], [0u64; 2]),
            |(e1, o1), (e2, o2)| {
                (
                    add_v(&Simd::from_array(e1), &Simd::from_array(e2)).to_array(),
                    add_v(&Simd::from_array(o1), &Simd::from_array(o2)).to_array(),
                )
            },
        );

    let sum0: Fp2SmallGoldilocks = unsafe { mem::transmute(sums.0) };
    let sum1: Fp2SmallGoldilocks = unsafe { mem::transmute(sums.1) };
    (sum0, sum1)
}

// has error
#[cfg(test)]
mod tests {
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;

    use super::super::MODULUS;
    use super::evaluate_ef;
    use crate::multilinear::pairwise;
    use crate::tests::{Fp2SmallGoldilocks, SmallGoldilocks};

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let src_bf: Vec<SmallGoldilocks> =
            (0..LEN).map(|_| SmallGoldilocks::rand(&mut rng)).collect();
        let src_ef: Vec<Fp2SmallGoldilocks> = src_bf
            .clone()
            .into_iter()
            .map(Fp2SmallGoldilocks::from_base_prime_field)
            .collect();

        // run function
        let expected_bf = pairwise::evaluate(&src_bf);
        let expected_ef = (
            Fp2SmallGoldilocks::from_base_prime_field(expected_bf.0),
            Fp2SmallGoldilocks::from_base_prime_field(expected_bf.1),
        );
        let received_ef = evaluate_ef::<MODULUS>(&src_ef);
        assert_eq!(expected_ef, received_ef);
    }
}
