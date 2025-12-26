use std::simd::{LaneCount, SupportedLaneCount};

use ark_std::{mem, simd::Simd, slice};
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, prelude::ParallelSlice};
use crate::{experimental::goldilocks::arithmetic::add::add_v, tests::Fp2SmallGoldilocks};

#[inline(always)]
fn sum_v<const LANES: usize>(src: &[u64]) -> ([u64; LANES], [u64; LANES])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut acc0 = Simd::<u64, LANES>::splat(0);
    let mut acc1 = Simd::<u64, LANES>::splat(0);
    
    let chunk_size = 2 * LANES; 
    let fast_path_end = src.len() - (src.len() % chunk_size);
    
    for i in (0..fast_path_end).step_by(chunk_size) {
        acc0 = add_v(&acc0, &Simd::<u64, LANES>::from_slice(&src[i..i + LANES]));
        acc1 = add_v(&acc1, &Simd::<u64, LANES>::from_slice(&src[i + LANES..i + 2 * LANES]));
    }

    // Final reduction from big SIMD registers to scalar Fp2 buckets
    let mut even_res = Simd::<u64, LANES>::splat(0); // [Real, Imag]
    let mut odd_res = Simd::<u64, LANES>::splat(0);  // [Real, Imag]

    let a0 = acc0.to_array();
    let a1 = acc1.to_array();
    
    for acc in &[a0, a1] {
        // We step by 4 because 1 Fp2 element = 2 lanes. 
        // 2 Fp2 elements (one even, one odd) = 4 lanes.
        for i in (0..LANES).step_by(4) {
            let e = Simd::<u64, LANES>::from_slice(&acc[i..i+2]);
            let o = Simd::<u64, LANES>::from_slice(&acc[i+2..i+4]);
            even_res = add_v(&even_res, &e);
            odd_res = add_v(&odd_res, &o);
        }
    }

    // // Handle trailing elements not processed by the main loop
    // for (i, chunk) in src[fast_path_end..].chunks_exact(2).enumerate() {
    //     let val = Simd::<u64, 2>::from_slice(chunk);
    //     if i % 2 == 0 {
    //         even_res = add_v(&even_res, &val);
    //     } else {
    //         odd_res = add_v(&odd_res, &val);
    //     }
    // }

    (even_res.to_array(), odd_res.to_array())
}

pub fn evaluate_ef<const MODULUS: u64>(src: &[Fp2SmallGoldilocks]) -> (Fp2SmallGoldilocks, Fp2SmallGoldilocks) {
    const CHUNK_SIZE: usize = 32_768;
    let (sum0_raw, sum1_raw): ([u64; 2], [u64; 2]) = src
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let chunk_raw: &[u64] = unsafe { 
                slice::from_raw_parts(chunk.as_ptr() as *const u64, chunk.len() * 2) 
            };
            // Use wide LANES for performance (16 is good for AVX2/AVX512)
            sum_v::<2>(chunk_raw) 
        })
        .reduce(
            || ([0u64; 2], [0u64; 2]),
            |(e1, o1), (e2, o2)| {
                (
                    add_v(&Simd::<u64, 2>::from_array(e1), &Simd::<u64, 2>::from_array(e2)).to_array(),
                    add_v(&Simd::<u64, 2>::from_array(o1), &Simd::<u64, 2>::from_array(o2)).to_array(),
                )
            },
        );

    let sum0: Fp2SmallGoldilocks = unsafe { mem::transmute(sum0_raw) };
    let sum1: Fp2SmallGoldilocks = unsafe { mem::transmute(sum1_raw) };
    (sum0, sum1)
}


// has error
#[cfg(test)]
mod tests {
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;

    use super::evaluate_ef;
    use crate::multilinear::pairwise;
    use crate::tests::{Fp2SmallGoldilocks, SmallGoldilocks};

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let src_bf: Vec<SmallGoldilocks> = (0..LEN).map(|_| SmallGoldilocks::rand(&mut rng)).collect();
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
        let received_ef = evaluate_ef::<18446744069414584321>(&src_ef);

        assert_eq!(expected_ef, received_ef);
    }
}