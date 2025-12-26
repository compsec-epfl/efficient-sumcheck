use ark_std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount};
use super::super::MODULUS;

#[inline(always)]
pub fn sub(a: u64, b: u64) -> u64 {
    let (diff, underflow) = a.overflowing_sub(b);
    if underflow {
        // If a < b, the raw diff is (a - b) + 2^64.
        // Since 2^64 mod p = 2^32 - 1, we subtract (2^32 - 1) to correct it.
        diff.wrapping_sub(0xFFFFFFFF)
    } else {
        diff
    }
}
#[inline(always)]
pub fn sub_v<const LANES: usize>(a: &Simd<u64, LANES>, b: &Simd<u64, LANES>) -> Simd<u64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let epsilon = Simd::<u64, LANES>::splat(0xFFFFFFFF);
    
    // 1. Standard wrapping subtraction
    let diff = *a - *b;
    
    // 2. Detect underflow (a < b)
    let underflow_mask = a.simd_lt(*b);
    
    // 3. If underflowed, we have diff = (a - b) + 2^64.
    // To get (a - b) mod p, we need (a - b) + (2^64 - 2^32 + 1).
    // So we subtract (2^32 - 1).
    underflow_mask.select(diff - epsilon, diff)
}

#[cfg(test)]
mod tests {
    use super::sub_v;
    use ark_std::simd::Simd;

    #[test]
    fn sanity() {
        let a: [u64; 1] = [9];
        let b: [u64; 1] = [7];
        let diff = sub_v(&Simd::from_array(a), &Simd::from_array(b));
        assert_eq!(diff[0], 2);
    }
}
