use ark_std::simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount};
use super::super::MODULUS;


// Why did not overflow in m31?
// M31 : You are using a u32. The maximum value of a + b is roughly 2^31 + 2^31 = 2^32. A u32 can hold up to 2^{32}-1. You barely fit, so it usually doesn't overflow in a way that triggers the panic (unless a or b are slightly above the modulus).

// Goldilocks (2^64 - 2^32 + 1): You are using a u64. The Goldilocks modulus is very close to the maximum value of a u64. If a and b are large field elements, their sum (a + b) will almost certainly exceed 2^64-1, causing the CPU to trigger an overflow panic before you even reach the if tmp >= MODULUS check.


// https://github.com/zhenfeizhang/Goldilocks/blob/872114997b82d0157e29a702992a3bd2023aa7ba/src/primefield/fp.rs#L377
#[inline(always)]
pub fn add(a: u64, b: u64) -> u64 {
    // We use wrapping_add to prevent the overflow panic.
    // sum = (a + b) mod 2^64
    let (sum, did_overflow) = a.overflowing_add(b);
    
    // If it overflowed 2^64, we need to add the "Goldilocks Epsilon" 
    // (which is 2^32 - 1) to account for the difference between 2^64 and the modulus.
    let epsilon = 0xFFFFFFFFu64;
    
    let mut res = if did_overflow {
        sum.wrapping_add(epsilon)
    } else {
        sum
    };

    // Final check to ensure the result is in [0, MODULUS)
    if res >= MODULUS {
        res -= MODULUS;
    }
    res
}

#[inline(always)]
pub fn add_v<const LANES: usize>(a: &Simd<u64, LANES>, b: &Simd<u64, LANES>) -> Simd<u64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let modulus = Simd::<u64, LANES>::splat(0xFFFFFFFF00000001);
    let epsilon = Simd::<u64, LANES>::splat(0xFFFFFFFF);

    // 1. Wrapping addition
    let sum = *a + *b;

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
