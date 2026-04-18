//! Generic field trait for sumcheck.
//!
//! [`SumcheckField`] captures the minimum arithmetic interface needed by the
//! sumcheck protocol. Any type with field-like operations (add, sub, mul,
//! negate, invert) and two distinguished constants (zero, one) can implement
//! this trait and use the full sumcheck library.
//!
//! When the `arkworks` feature is enabled (default), a blanket implementation
//! is provided for all types implementing [`ark_ff::Field`], so existing
//! arkworks users change nothing.

use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Minimum field interface for the sumcheck protocol.
///
/// Implementors must provide:
/// - Standard arithmetic via [`Add`], [`Sub`], [`Mul`], [`Neg`] and their
///   assign variants.
/// - Additive and multiplicative identities ([`ZERO`](Self::ZERO),
///   [`ONE`](Self::ONE)).
/// - Conversion from small integers ([`from_u64`](Self::from_u64)).
/// - Multiplicative inverse ([`inverse`](Self::inverse)).
///
/// The SIMD acceleration layer for Goldilocks (p = 2^64 − 2^32 + 1) is
/// opt-in via [`_simd_field_config`](Self::_simd_field_config). Non-Goldilocks
/// fields leave the default (returns `None`) and the library transparently
/// falls back to scalar code.
pub trait SumcheckField:
    Sized
    + Copy
    + Send
    + Sync
    + PartialEq
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Sum
    + 'static
{
    /// Additive identity.
    const ZERO: Self;

    /// Multiplicative identity.
    const ONE: Self;

    /// Convert a small integer to a field element.
    fn from_u64(val: u64) -> Self;

    /// Multiplicative inverse, or `None` for zero.
    fn inverse(&self) -> Option<Self>;

    /// Returns `true` if this element is zero.
    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// Double this element (additive).
    #[inline]
    fn double(&self) -> Self {
        *self + *self
    }

    /// Extension degree over the base prime field.
    ///
    /// For a prime field, return 1. For a quadratic extension, return 2, etc.
    /// Used by the SIMD layer to select the correct kernel width.
    fn extension_degree() -> u64 {
        1
    }

    /// SIMD configuration hint (crate-internal, not part of the public API).
    ///
    /// Returns `Some(config)` if this field's elements can be processed by
    /// the SIMD acceleration layer. The default returns `None` (no SIMD).
    ///
    /// For Goldilocks base field: return `Some(SimdFieldConfig { modulus: 0xFFFF_FFFF_0000_0001, element_bytes: 8 })`.
    /// For Goldilocks extensions: the base prime field's config is used.
    #[doc(hidden)]
    #[inline(always)]
    fn _simd_field_config() -> Option<SimdFieldConfig> {
        None
    }
}

/// SIMD field configuration hint.
///
/// Returned by [`SumcheckField::_simd_field_config`] to tell the SIMD
/// dispatch layer how to handle this field's elements.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdFieldConfig {
    /// The prime field modulus as a single `u64` limb.
    /// Only meaningful when `element_bytes == 8` (single-limb field).
    pub modulus: u64,

    /// Size of one base prime field element in bytes.
    pub element_bytes: usize,
}

/// Marker trait for cross-field (base field → extension field) sumcheck.
///
/// The prover starts with evaluations in `BF` and folds into `EF` once
/// the first challenge arrives.
pub trait ExtensionOf<BF: SumcheckField>: SumcheckField + From<BF> {}

// ─── Arkworks blanket implementation ────────────────────────────────────────

#[cfg(feature = "arkworks")]
mod ark_impl {
    use super::*;

    impl<F> SumcheckField for F
    where
        F: ark_ff::Field,
    {
        const ZERO: Self = <Self as ark_ff::AdditiveGroup>::ZERO;
        const ONE: Self = <Self as ark_ff::Field>::ONE;

        #[inline]
        fn from_u64(val: u64) -> Self {
            Self::from(val)
        }

        #[inline]
        fn inverse(&self) -> Option<Self> {
            ark_ff::Field::inverse(self)
        }

        #[inline]
        fn is_zero(&self) -> bool {
            *self == Self::ZERO
        }

        #[inline]
        fn double(&self) -> Self {
            ark_ff::AdditiveGroup::double(self)
        }

        #[inline]
        fn extension_degree() -> u64 {
            <Self as ark_ff::Field>::extension_degree()
        }

        #[inline(always)]
        fn _simd_field_config() -> Option<SimdFieldConfig> {
            use ark_ff::PrimeField;

            // Check if the base prime field is a single u64 (64-bit modulus).
            if F::BasePrimeField::MODULUS_BIT_SIZE != 64 {
                return None;
            }
            let d = <Self as ark_ff::Field>::extension_degree() as usize;
            if core::mem::size_of::<F>() != d * 8 {
                return None;
            }
            let modulus = F::BasePrimeField::MODULUS;
            let limbs: &[u64] = modulus.as_ref();
            if limbs[1..].iter().any(|&x| x != 0) {
                return None;
            }
            Some(SimdFieldConfig {
                modulus: limbs[0],
                element_bytes: 8,
            })
        }
    }

    // Blanket ExtensionOf for arkworks extension fields.
    impl<BF, EF> ExtensionOf<BF> for EF
    where
        BF: ark_ff::Field,
        EF: ark_ff::Field + From<BF>,
    {
    }
}
