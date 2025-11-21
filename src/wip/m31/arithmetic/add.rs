use ark_std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount};

const MODULUS: u32 = 2_147_483_647;

#[inline(always)]
pub fn add(a: u32, b: u32) -> u32 {
    let tmp = a + b;
    if tmp >= MODULUS {
        tmp - MODULUS
    } else {
        tmp
    }
}

#[inline(always)]
pub fn add_v<const LANES: usize>(a: &Simd<u32, LANES>, b: &Simd<u32, LANES>) -> Simd<u32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // TODO (z-tech): can this be made const?
    let modulus = Simd::<u32, LANES>::splat(MODULUS);
    let sum = a + b;
    sum.simd_ge(modulus).select(sum - modulus, sum)
}
