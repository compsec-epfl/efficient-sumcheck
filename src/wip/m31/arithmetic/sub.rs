use ark_std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn sub_v<const LANES: usize>(
    a: Simd<u32, LANES>,
    b: Simd<u32, LANES>,
    modulus: &Simd<u32, LANES>,
) -> Simd<u32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let diff = a - b;
    a.simd_lt(b).select(diff + modulus, diff)
}
