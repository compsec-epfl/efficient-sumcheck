use ark_std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn add_v<const LANES: usize>(
    a: &Simd<u32, LANES>,
    b: &Simd<u32, LANES>,
    modulus: &Simd<u32, LANES>,
) -> Simd<u32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let sum = a + b;
    sum.simd_ge(*modulus).select(sum - modulus, sum)
}
