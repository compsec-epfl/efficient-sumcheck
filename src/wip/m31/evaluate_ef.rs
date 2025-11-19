use ark_std::{
    mem,
    simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount},
    slice,
    slice::from_raw_parts,
};
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, prelude::ParallelSlice};

#[cfg(feature = "parallel")]
use crate::wip::m31::vectorized_reductions::pairwise::add_fp4_raw;

use crate::tests::Fp4SmallM31;

fn is_serial_better(len: usize, break_even_len: usize) -> bool {
    len < break_even_len
}

#[inline(always)]
fn sum_assign<const LANES: usize, const MODULUS: u32>(
    a: &mut Simd<u32, LANES>,
    b: &Simd<u32, LANES>,
    modulus: &Simd<u32, LANES>,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    *a += b;
    *a = a.simd_ge(*modulus).select(*a - modulus, *a);
}

fn reduce_sum_packed_ef<const MODULUS: u32>(src: &[u32]) -> ([u32; 4], [u32; 4]) {
    // Must be list of Fp4SmallM31s in memory like: &[a0, a1, a2, a3, b0, b1, b2, b3]
    // and they must come in pairs of two
    assert!(src.len().is_multiple_of(8));

    let modulus = Simd::<u32, 4>::splat(MODULUS);
    let mut acc0: Simd<u32, 4> = Simd::<u32, 4>::splat(0);
    let mut acc1: Simd<u32, 4> = Simd::<u32, 4>::splat(0);
    let mut acc2: Simd<u32, 4> = Simd::<u32, 4>::splat(0);
    let mut acc3: Simd<u32, 4> = Simd::<u32, 4>::splat(0);

    let len = src.len();
    let chunk_size = 4 * 4;
    for i in (0..len).step_by(chunk_size) {
        sum_assign::<4, MODULUS>(
            &mut acc0,
            &Simd::<u32, 4>::from_slice(&src[i..i + 4]),
            &modulus,
        );
        sum_assign::<4, MODULUS>(
            &mut acc1,
            &Simd::<u32, 4>::from_slice(&src[i + 4..i + 2 * 4]),
            &modulus,
        );
        sum_assign::<4, MODULUS>(
            &mut acc2,
            &Simd::<u32, 4>::from_slice(&src[i + 2 * 4..i + 3 * 4]),
            &modulus,
        );
        sum_assign::<4, MODULUS>(
            &mut acc3,
            &Simd::<u32, 4>::from_slice(&src[i + 3 * 4..i + 4 * 4]),
            &modulus,
        );
    }

    // one more reduction
    sum_assign::<4, MODULUS>(&mut acc0, &acc2, &modulus);
    sum_assign::<4, MODULUS>(&mut acc1, &acc3, &modulus);

    let sums = (*acc0.as_array(), *acc1.as_array());
    sums
}

pub fn evaluate_ef<const LANES: usize, const MODULUS: u32>(
    src: &[Fp4SmallM31],
) -> (Fp4SmallM31, Fp4SmallM31) {
    // TODO (z-tech): break even is machine dependent
    if is_serial_better(src.len(), 1 << 16) || !cfg!(feature = "parallel") {
        let src_raw: &[u32] = unsafe { from_raw_parts(src.as_ptr() as *const u32, src.len() * 4) };
        let (sum0_raw, sum1_raw) = reduce_sum_packed_ef::<MODULUS>(src_raw);
        return (
            unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(sum0_raw) },
            unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(sum1_raw) },
        );
    }

    // --- parallel path ---
    #[cfg(feature = "parallel")]
    {
        let n_threads = rayon::current_num_threads();
        let chunk_size = src.len() / n_threads;
        let sums = src
            .par_chunks(chunk_size)
            .map(|chunk| {
                let chunk_raw: &[u32] =
                    unsafe { slice::from_raw_parts(chunk.as_ptr() as *const u32, chunk.len() * 4) };
                reduce_sum_packed_ef::<MODULUS>(chunk_raw)
            })
            .reduce(
                || ([0u32; 4], [0u32; 4]),
                |(e1, o1), (e2, o2)| (add_fp4_raw(e1, e2), add_fp4_raw(o1, o2)),
            );

        let (sum0_raw, sum1_raw) = sums;
        let sum0: Fp4SmallM31 = unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(sum0_raw) };
        let sum1: Fp4SmallM31 = unsafe { mem::transmute::<[u32; 4], Fp4SmallM31>(sum1_raw) };

        (sum0, sum1)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;

    use crate::multilinear::pairwise;
    use crate::tests::{Fp4SmallM31, SmallM31};
    use crate::wip::m31::evaluate_ef::evaluate_ef;

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
        let received_4_lanes_ef = evaluate_ef::<4, 2_147_483_647>(&src_ef);

        assert_eq!(expected_ef, received_4_lanes_ef);

        let received_8_lanes_ef = evaluate_ef::<8, 2_147_483_647>(&src_ef);
        assert_eq!(expected_ef, received_8_lanes_ef);
    }
}
