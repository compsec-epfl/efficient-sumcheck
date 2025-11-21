use ark_std::simd::{cmp::SimdPartialOrd, num::SimdUint, LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn mul_v<const LANES: usize>(a: &Simd<u32, LANES>, b: &Simd<u32, LANES>) -> Simd<u32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // widen
    let mut widend_a: Simd<u64, LANES> = a.cast();
    let widend_b: Simd<u64, LANES> = b.cast();

    // mul
    widend_a *= widend_b;

    // mersenne reduction
    let modulus = Simd::<u64, LANES>::splat(2_147_483_647); // const?
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
    use crate::wip::m31::arithmetic::mul::mul_v;
    use ark_std::{rand::RngCore, simd::Simd, test_rng};

    #[test]
    fn sanity() {
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let multipliers: Vec<u32> = (0..LEN).map(|_| rng.next_u32() % 2_147_483_647).collect();
        let mut expected_ef: Vec<u32> = (0..LEN).map(|_| rng.next_u32()).collect();
        let mut received_ef = expected_ef.clone();

        // control
        expected_ef
            .iter_mut()
            .zip(multipliers.iter())
            .for_each(|(a, b)| {
                let prod = (*a as u64) * (*b as u64);
                *a = (prod % 2_147_483_647) as u32;
            });

        const LANES: usize = 16;
        for (a_chunk, b_chunk) in received_ef.chunks_mut(LANES).zip(multipliers.chunks(LANES)) {
            let a_simd = Simd::<u32, LANES>::from_slice(a_chunk);
            let b_simd = Simd::<u32, LANES>::from_slice(b_chunk);

            // perfom op
            let res = mul_v(&a_simd, &b_simd);

            // write back into slice
            a_chunk.copy_from_slice(res.as_array());
        }

        assert_eq!(expected_ef, received_ef);
    }
}
