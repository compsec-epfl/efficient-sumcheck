use ark_std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount};

const MODULUS: u32 = 2_147_483_647;

#[inline(always)]
pub fn sub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + MODULUS - b
    }
}

#[inline(always)]
pub fn sub_v<const LANES: usize>(a: &Simd<u32, LANES>, b: &Simd<u32, LANES>) -> Simd<u32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let modulus = Simd::<u32, LANES>::splat(MODULUS); // const?
    let diff = a - b;
    a.simd_lt(*b).select(diff + modulus, diff)
}

#[cfg(test)]
mod tests {
    use crate::experimental::m31::arithmetic::sub::sub_v;
    use ark_std::simd::Simd;

    #[test]
    fn sanity() {
        let a: [u32; 1] = [9];
        let b: [u32; 1] = [7];
        let diff = sub_v(&Simd::from_array(a), &Simd::from_array(b));
        assert_eq!(diff[0], 2);
    }
}
