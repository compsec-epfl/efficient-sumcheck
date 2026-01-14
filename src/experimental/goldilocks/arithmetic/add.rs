use super::super::{EPSILON, MODULUS};
use crate::experimental::goldilocks::utils::{assume, branch_hint};
use ark_std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount};

// https://github.com/zhenfeizhang/Goldilocks/blob/872114997b82d0157e29a702992a3bd2023aa7ba/src/primefield/fp.rs#L377
#[inline(always)]
pub fn add(a: u64, b: u64) -> u64 {
    let (sum, over) = a.overflowing_add(b);
    let (mut sum, over) = sum.overflowing_add((over as u64) * EPSILON);
    if over {
        // NB: a > Self::ORDER && b > Self::ORDER is necessary but not sufficient for double-overflow.
        // This assume does two things:
        //  1. If compiler knows that either a or b <= ORDER, then it can skip this check.
        //  2. Hints to the compiler how rare this double-overflow is (thus handled better with a branch).
        assume(a > MODULUS && b > MODULUS);
        branch_hint();
        sum += EPSILON; // Cannot overflow.
    }
    sum
}

#[inline(always)]
pub fn add_v<const LANES: usize>(a: &Simd<u64, LANES>, b: &Simd<u64, LANES>) -> Simd<u64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let modulus = Simd::<u64, LANES>::splat(MODULUS);
    let epsilon = Simd::<u64, LANES>::splat(EPSILON);
    let sum = a + b;

    // 2. Detect where overflow occurred (a + b >= 2^64)
    // In SIMD, if the sum is less than one of the inputs, an overflow happened.
    let overflow_mask = sum.simd_lt(*a);

    // 3. Add epsilon to lanes that overflowed
    let mut res = overflow_mask.select(sum + epsilon, sum);

    // 4. Final canonical reduction: if res >= modulus { res - modulus }
    res = res.simd_ge(modulus).select(res - modulus, res);

    res
}

#[cfg(test)]
mod tests {
    use super::add_v;
    use ark_std::simd::Simd;

    #[test]
    fn sanity() {
        let a: [u64; 1] = [9];
        let b: [u64; 1] = [7];
        let sum = add_v(&Simd::from_array(a), &Simd::from_array(b));
        assert_eq!(sum[0], 16);
    }
}
