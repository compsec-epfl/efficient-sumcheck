use core::option::Option::None;

use crate::fields::m31::{M31, M31_MODULUS};
use ark_ff::{Field, LegendreSymbol, One, PrimeField, Zero};
use ark_serialize::Flags;

// TODO (z-tech): Each of these needs to implemented w/ tests

impl Field for M31 {
    type BasePrimeField = Self;

    type BasePrimeFieldIter = ark_std::iter::Once<Self::BasePrimeField>;

    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;

    const ZERO: Self = Self { value: 0 };

    const ONE: Self = Self { value: 1 };

    fn double(&self) -> Self {
        M31::from((2 * self.value) % M31_MODULUS)
    }

    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let x = *self;
        let y = x.exp_power_of_2(2) * x;
        let z = y.square() * y;
        let a = z.exp_power_of_2(4) * z;
        let b = a.exp_power_of_2(4);
        let c = b * z;
        let d = b.exp_power_of_2(4) * a;
        let e = d.exp_power_of_2(12) * c;
        let f = e.exp_power_of_2(3) * y;
        Some(f)
    }

    fn frobenius_map(&self, _: usize) -> M31 {
        Self { value: self.value }
    }

    // M31 is prime field
    fn extension_degree() -> u64 {
        1
    }

    // `Field`` is designed to support prime fields and extension fields. The
    // iterator is needed for `m` extension field, where a signle element
    // consists `m` elements of the prime field. However for prime fields this
    // becomes trivial.
    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
        ark_std::iter::once(*self)
    }

    fn from_base_prime_field_elems(_elems: &[Self::BasePrimeField]) -> Option<Self> {
        if _elems.len() != (Self::extension_degree() as usize) {
            return None;
        }
        Some(_elems[0])
    }

    // Base prime field is the same as the prime field the element does not need to be
    // changed at all
    fn from_base_prime_field(_elem: Self::BasePrimeField) -> Self {
        _elem
    }

    fn double_in_place(&mut self) -> &mut Self {
        *self += *self;
        self
    }

    fn neg_in_place(&mut self) -> &mut Self {
        if !self.is_zero() {
            *self = -*self;
        }
        self
    }

    // Takes the first 4 bytes into u32 and the 5th byte is a flag
    fn from_random_bytes_with_flags<F: Flags>(_bytes: &[u8]) -> Option<(Self, F)> {
        if _bytes.len() < 4 {
            return None;
        }
        let mut array = [0u8; 4];
        array.copy_from_slice(&_bytes[..4]);
        let field_element = M31::from(u32::from_le_bytes(array));

        // ! Check what is the purpose of flags
        let flags = F::from_u8(_bytes.get(4).copied().unwrap_or(0))?;
        Some((field_element, flags))
    }

    // To determine if an element a is a quadratic residue use Eurler's criterion
    // an element a is a quadratic residue if a^{(p-1)/2} = 1
    fn legendre(&self) -> ark_ff::LegendreSymbol {
        if self.is_zero() {
            return LegendreSymbol::Zero;
        }
        if self.pow(&<Self>::MODULUS_MINUS_ONE_DIV_TWO.0).is_one() {
            return LegendreSymbol::QuadraticResidue;
        }
        LegendreSymbol::QuadraticNonResidue
    }

    fn square(&self) -> Self {
        self.clone() * self.clone()
    }

    fn square_in_place(&mut self) -> &mut Self {
        *self *= *self;
        self
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        *self = self.inverse()?;
        Some(self)
    }

    // The map should be a -> a^p. By Fermat a^{p-1} = 1 thus a^p = a
    fn frobenius_map_in_place(&mut self, _power: usize) {}

    fn characteristic() -> &'static [u64] {
        &[M31_MODULUS as u64]
    }

    fn from_random_bytes(_bytes: &[u8]) -> Option<Self> {
        if _bytes.len() < 4 {
            return None;
        }
        let mut array = [0u8; 4];
        array.copy_from_slice(&_bytes[..4]);
        let field_element = M31::from(u32::from_le_bytes(array));
        Some(field_element)
    }

    // Consider an element a = x^2, then from the Euler's criterion we know
    // a^{(p-1)/2} = 1.
    // Notice a^{(p+1)/2} = a^{(p-1)/2}*a = a which implies
    // a^{{(p+1)/4}^2} = a thus a^{{(p+1)/4}} is the square root
    fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::zero());
        }

        if self.legendre() != LegendreSymbol::QuadraticResidue {
            return None;
        }

        const EXP: u64 = (M31_MODULUS as u64 + 1) / 4;
        let root = self.pow(&[EXP]);
        Some(root)
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        *self = self.sqrt()?;
        Some(self)
    }

    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        let mut sum = Self::zero();
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    fn pow<S: AsRef<[u64]>>(&self, _exp: S) -> Self {
        let base = *self;
        let mut result = Self::ONE;
        let mut exp = _exp
            .as_ref()
            .iter()
            .flat_map(|&x| (0..64).map(move |i| (x >> i) & 1));

        while let Some(bit) = exp.next() {
            result = result.square();
            if bit == 1 {
                result *= base;
            }
        }
        result
    }

    fn pow_with_table<S: AsRef<[u64]>>(_powers_of_2: &[Self], _exp: S) -> Option<Self> {
        std::unimplemented!()
    }
}

// Randomized test???
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double() {
        let a = M31::from(10);
        let result = a.double();
        assert_eq!(result, M31::from((10 * 2) % M31_MODULUS));

        let b = M31::from(M31_MODULUS - 1);
        let result = b.double();
        assert_eq!(result, M31::from((2 * (M31_MODULUS - 1)) % M31_MODULUS));
    }

    #[test]
    fn test_inverse() {
        let a = M31::from(3);
        let inv_a = a.inverse().unwrap();
        assert_eq!(a * inv_a, M31::ONE);

        let zero = M31::ZERO;
        assert!(zero.inverse().is_none());
    }

    #[test]
    fn test_to_base_prime_field_elements() {
        let a = M31::from(123);
        let elems: Vec<M31> = a.to_base_prime_field_elements().collect();
        assert_eq!(elems, vec![a]);
    }

    #[test]
    fn test_pow_zero() {
        let a = M31::from(42u64);
        // exponent = 0
        assert_eq!(a.pow(&[0]), M31::ONE);
    }
}
