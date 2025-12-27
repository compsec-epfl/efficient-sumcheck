use std::simd::num::SimdUint;

use ark_std::{
    mem,
    simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount},
};

use crate::tests::SmallGoldilocks;
pub const MODULUS: u64 = 0xffffffff00000001;

#[inline(always)]
pub fn mul(a: u64, b: u64) -> u64 {
    let prod = unsafe { mem::transmute::<u64, SmallGoldilocks>(a) }
        * unsafe { mem::transmute::<u64, SmallGoldilocks>(b) };
    unsafe { mem::transmute::<SmallGoldilocks, u64>(prod) }
}

#[inline(always)]
pub fn mul_v<const LANES: usize>(
    a: &Simd<u64, LANES>,
    b: &Simd<u64, LANES>,
) -> Simd<u64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // For goldilocks Simd<u128, LANES> dont exist, look for other ways
    // widen
    let mut widend_a: Simd<u64, LANES> = a.cast();
    let widend_b: Simd<u64, LANES> = b.cast();

    // mul
    widend_a *= widend_b;

    // TODO (z-tech): can this be made const?
    let modulus = Simd::<u64, LANES>::splat(MODULUS);

    // mersenne reduction
    let low = widend_a & modulus;
    let high = widend_a >> Simd::<u64, LANES>::splat(31);
    let mut reduced = low + high;

    // At most 2 subtractions needed
    reduced = reduced.simd_ge(modulus).select(reduced - modulus, reduced);
    reduced = reduced.simd_ge(modulus).select(reduced - modulus, reduced);

    // done
    reduced.cast()
}


#[cfg(test)]
mod tests {
    use crate::experimental::goldilocks::MODULUS;

    use super::mul_v;
    use ark_std::{rand::RngCore, simd::Simd, test_rng};

    #[test]
    fn sanity() {
        const LEN: usize = 16;
        let mut rng = test_rng();

        // random elements
        let multipliers: Vec<u64> = (0..LEN).map(|_| rng.next_u64() % MODULUS).collect();
        // println!("{:?}", multipliers);

        let mut expected_ef: Vec<u64> = (0..LEN).map(|_| rng.next_u64()).collect();
        // println!("{:?}", expected_ef);

        let mut received_ef = expected_ef.clone();

        // control
        expected_ef
            .iter_mut()
            .zip(multipliers.iter())
            .for_each(|(a, b)| {
                let prod = (*a as u128) * (*b as u128);
                *a = (prod % 0xffffffff00000001) as u64;
            });

        const LANES: usize = 16;
        for (a_chunk, b_chunk) in received_ef.chunks_mut(LANES).zip(multipliers.chunks(LANES)) {
            let a_simd = Simd::<u64, LANES>::from_slice(a_chunk);
            let b_simd = Simd::<u64, LANES>::from_slice(b_chunk);

            // perfom op
            let res = mul_v(&a_simd, &b_simd);

            // write back into slice
            a_chunk.copy_from_slice(res.as_array());
        }

        assert_eq!(expected_ef, received_ef);
    }
}
