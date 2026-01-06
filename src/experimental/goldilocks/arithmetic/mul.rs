use std::simd::{Mask};

use ark_std::{
    mem,
    simd::{cmp::SimdPartialOrd, LaneCount, Simd, SupportedLaneCount},
};

use crate::tests::SmallGoldilocks;
use super::super::{MODULUS, EPSILON};

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
    let mask32 = Simd::splat(0xFFFFFFFFu64);

    let a_lo = *a & mask32;
    let a_hi = *a >> 32;
    let b_lo = *b & mask32;
    let b_hi = *b >> 32;

    let lo_lo = a_lo * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_lo = a_hi * b_lo;
    let hi_hi = a_hi * b_hi;

    let mid = lo_hi + hi_lo;
    let mid_carry = mid.simd_lt(lo_hi).select(Simd::splat(1 << 32), Simd::splat(0));
    
    let mid_lo = mid & mask32;
    let mid_hi = mid >> 32;

    let x_lo = lo_lo + (mid_lo << 32);
    let x_lo_carry = x_lo.simd_lt(lo_lo).select(Simd::splat(1), Simd::splat(0));
    let x_hi = hi_hi + mid_hi + mid_carry + x_lo_carry;

    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & mask32;


    let mut t0 = x_lo - x_hi_hi;
    let borrow_mask = x_lo.simd_lt(x_hi_hi);
    t0 = borrow_mask.select(t0 - Simd::splat(EPSILON), t0);


    let t1 = x_hi_lo * Simd::splat(EPSILON);

    let (t2_wrapped, carry) = overflowing_add_simd(t0, t1);
    let mut r = t2_wrapped + (carry.select(Simd::splat(EPSILON), Simd::splat(0)));

    let p = Simd::splat(MODULUS);
    r = r.simd_ge(p).select(r - p, r);

    r
}

/// Helper for overflowing add in SIMD
#[inline(always)]
fn overflowing_add_simd<const LANES: usize>(
    a: Simd<u64, LANES>, 
    b: Simd<u64, LANES>
) -> (Simd<u64, LANES>, Mask<i64, LANES>) 
where LaneCount<LANES>: SupportedLaneCount 
{
    let res = a + b;
    (res, res.simd_lt(a))
}


#[cfg(test)]
mod tests {
    use crate::experimental::goldilocks::MODULUS;

    use super::mul_v;
    use ark_std::{rand::RngCore, simd::Simd, test_rng};

     #[test]
    fn single() {        
        // https://asecuritysite.com/zk/go_plonk4

        let a_input: [u64; 1] = [10719222850664546238];
        let b_input: [u64; 1] = [301075827032876239];

        // 1. Calculate Expected using u128
        let expected = ((a_input[0] as u128 * b_input[0] as u128) % MODULUS as u128) as u64;

        // 2. Calculate Received using your mul_v
        const LANES: usize = 1;
        let a_simd = Simd::<u64, LANES>::from_slice(&a_input);
        let b_simd = Simd::<u64, LANES>::from_slice(&b_input);
        let res_simd = mul_v(&a_simd, &b_simd);
        let received = res_simd.as_array()[0];

        println!("Expected: {}, Received: {}", expected, received);
        assert_eq!(expected, received);
    }

    #[test]
    fn sanity() {
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let multipliers: Vec<u64> = (0..LEN).map(|_| rng.next_u64() % MODULUS).collect();
        let mut expected_ef: Vec<u64> = (0..LEN).map(|_| rng.next_u64()).collect();

        let mut received_ef = expected_ef.clone();

        // control
        expected_ef
            .iter_mut()
            .zip(multipliers.iter())
            .for_each(|(a, b)| {
                let prod = (*a as u128) * (*b as u128);
                *a = (prod % MODULUS as u128) as u64;
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
