//! `SimdAccelerated` implementation for Goldilocks (`F64`).
//!
//! Uses the Montgomery-form NEON backend (`MontGoldilocksNeon`) which operates
//! directly on arkworks' internal representation — zero-cost access,
//! no conversion needed.

use super::MontGoldilocksSIMD;
use crate::simd_fields::SimdAccelerated;
use crate::tests::F64;

impl SimdAccelerated for F64 {
    type Backend = MontGoldilocksSIMD;

    #[inline(always)]
    fn to_raw(val: F64) -> u64 {
        // SmallFp { value: u64, _phantom } — direct access to Montgomery-form value.
        val.value
    }

    #[inline(always)]
    fn from_raw(val: u64) -> F64 {
        // Construct SmallFp directly from Montgomery-form value (no conversion).
        F64::from_raw(val)
    }

    #[inline(always)]
    fn slice_to_raw(src: &[F64]) -> Vec<u64> {
        // Zero-cost: SmallFp<P> is repr-compatible with u64 (value: u64 + ZST PhantomData).
        let mut out = Vec::with_capacity(src.len());
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr() as *const u64, out.as_mut_ptr(), src.len());
            out.set_len(src.len());
        }
        out
    }

    #[inline(always)]
    fn slice_from_raw(src: &[u64]) -> Vec<F64> {
        let mut out = Vec::with_capacity(src.len());
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr() as *const F64, out.as_mut_ptr(), src.len());
            out.set_len(src.len());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd_fields::SimdBaseField;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_roundtrip() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let f = F64::rand(&mut rng);
            let raw = <F64 as SimdAccelerated>::to_raw(f);
            let back = <F64 as SimdAccelerated>::from_raw(raw);
            assert_eq!(f, back);
        }
    }

    #[test]
    fn test_slice_roundtrip() {
        let mut rng = test_rng();
        let n = 1024;
        let original: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let raw = <F64 as SimdAccelerated>::slice_to_raw(&original);
        let recovered = <F64 as SimdAccelerated>::slice_from_raw(&raw);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_arithmetic_in_mont_domain() {
        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);

            // Add
            let expected_sum = a + b;
            let raw_sum = MontGoldilocksSIMD::scalar_add(
                <F64 as SimdAccelerated>::to_raw(a),
                <F64 as SimdAccelerated>::to_raw(b),
            );
            assert_eq!(
                <F64 as SimdAccelerated>::from_raw(raw_sum),
                expected_sum,
                "add mismatch"
            );

            // Mul (Montgomery mul in the raw domain should match arkworks mul)
            let expected_prod = a * b;
            let raw_prod = MontGoldilocksSIMD::scalar_mul(
                <F64 as SimdAccelerated>::to_raw(a),
                <F64 as SimdAccelerated>::to_raw(b),
            );
            assert_eq!(
                <F64 as SimdAccelerated>::from_raw(raw_prod),
                expected_prod,
                "mul mismatch"
            );
        }
    }
}
