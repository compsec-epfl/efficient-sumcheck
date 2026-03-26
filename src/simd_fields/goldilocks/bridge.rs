//! `SimdAccelerated` implementation for Goldilocks (`F64`).
//!
//! Bridges the arkworks `Fp64<MontBackend<F64Config, 1>>` type to the
//! [`GoldilocksSIMD`] backend by converting between Montgomery and canonical form.

use ark_ff::PrimeField;

use super::GoldilocksSIMD;
use crate::simd_fields::SimdAccelerated;
use crate::tests::F64;

impl SimdAccelerated for F64 {
    type Backend = GoldilocksSIMD;

    #[inline]
    fn to_raw(val: F64) -> u64 {
        // into_bigint() converts from Montgomery form to canonical
        val.into_bigint().0[0]
    }

    #[inline]
    fn from_raw(val: u64) -> F64 {
        F64::from_bigint(ark_ff::BigInt([val])).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_arithmetic_in_raw_domain() {
        use crate::simd_fields::goldilocks::GoldilocksSIMD;
        use crate::simd_fields::SimdBaseField;

        let mut rng = test_rng();
        for _ in 0..10_000 {
            let a = F64::rand(&mut rng);
            let b = F64::rand(&mut rng);

            // Add
            let ff_sum = a + b;
            let raw_sum = GoldilocksSIMD::scalar_add(
                <F64 as SimdAccelerated>::to_raw(a),
                <F64 as SimdAccelerated>::to_raw(b),
            );
            assert_eq!(
                <F64 as SimdAccelerated>::to_raw(ff_sum),
                raw_sum,
                "add mismatch"
            );

            // Mul
            let ff_prod = a * b;
            let raw_prod = GoldilocksSIMD::scalar_mul(
                <F64 as SimdAccelerated>::to_raw(a),
                <F64 as SimdAccelerated>::to_raw(b),
            );
            assert_eq!(
                <F64 as SimdAccelerated>::to_raw(ff_prod),
                raw_prod,
                "mul mismatch"
            );
        }
    }
}
