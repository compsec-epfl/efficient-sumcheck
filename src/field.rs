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

    /// SIMD configuration (internal dispatch hook).
    ///
    /// For **arkworks types**: overridden by the blanket impl to auto-detect
    /// Goldilocks from the modulus. Const-folded by LLVM.
    ///
    /// For **non-arkworks types**: override this to return the config from
    /// your [`SimdRepr`] implementation. Or simply implement `SimdRepr` and
    /// override this method to call [`simd_config_from_repr`]:
    ///
    /// ```ignore
    /// fn _simd_field_config() -> Option<SimdFieldConfig> {
    ///     Some(simd_config_from_repr::<Self>())
    /// }
    /// ```
    #[doc(hidden)]
    #[inline(always)]
    fn _simd_field_config() -> Option<SimdFieldConfig> {
        None
    }

    /// Reinterpret a field element as its raw Montgomery-form `u64`.
    ///
    /// Only meaningful for Goldilocks-based fields where each base-field
    /// component is a single `u64` in Montgomery form. The default panics;
    /// overridden by the arkworks blanket impl and by `SimdRepr` types.
    #[doc(hidden)]
    #[inline(always)]
    fn _to_raw_u64(self) -> u64 {
        unimplemented!("_to_raw_u64 is only available for SIMD-compatible fields")
    }

    /// Reconstruct a field element from a raw Montgomery-form `u64`.
    #[doc(hidden)]
    #[inline(always)]
    fn _from_raw_u64(raw: u64) -> Self {
        let _ = raw;
        unimplemented!("_from_raw_u64 is only available for SIMD-compatible fields")
    }

    /// Reinterpret a slice of field elements as a flat `u64` slice.
    ///
    /// For base fields (degree 1), the output has the same length as input.
    /// For extension fields (degree d), the output has `input.len() * d` elements.
    #[doc(hidden)]
    #[inline(always)]
    fn _as_u64_slice(slice: &[Self]) -> &[u64] {
        let _ = slice;
        unimplemented!("_as_u64_slice is only available for SIMD-compatible fields")
    }

    /// Reinterpret a mutable slice of field elements as a mutable flat `u64` slice.
    #[doc(hidden)]
    #[inline(always)]
    fn _as_u64_slice_mut(slice: &mut [Self]) -> &mut [u64] {
        let _ = slice;
        unimplemented!("_as_u64_slice_mut is only available for SIMD-compatible fields")
    }

    /// Reconstruct a field element from its raw Montgomery-form `u64` components.
    ///
    /// For base fields, `comps` has length 1. For degree-d extensions, length d.
    #[doc(hidden)]
    #[inline(always)]
    fn _from_u64_components(comps: &[u64]) -> Self {
        let _ = comps;
        unimplemented!("_from_u64_components is only available for SIMD-compatible fields")
    }
}

// ─── SIMD memory layout contract ────────────────────────────────────────────

/// Goldilocks modulus: p = 2^64 - 2^32 + 1.
pub const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Opt-in trait for SIMD acceleration.
///
/// Implementing this trait declares that the field type's in-memory
/// representation is compatible with the SIMD kernels: each element is
/// `extension_degree()` consecutive `u64` values in Montgomery form.
///
/// # Layout safety
///
/// The memory layout guarantee is enforced by the **zerocopy** bounds
/// (`IntoBytes + FromBytes + Immutable`). These are verified at compile
/// time by zerocopy's derive macros — no `unsafe` needed from the
/// implementor. If your type's layout doesn't support safe byte
/// reinterpretation, the derive will fail to compile.
///
/// The only thing the implementor declares is the **modulus value**.
/// A wrong modulus produces wrong arithmetic results (logic bug), not
/// undefined behavior.
///
/// # Example: non-arkworks Goldilocks
///
/// ```ignore
/// #[derive(Clone, Copy, Debug, PartialEq,
///          zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
/// #[repr(transparent)]
/// struct MyGoldilocks(u64);
///
/// impl SumcheckField for MyGoldilocks {
///     // ... arithmetic ...
///     fn _simd_field_config() -> Option<SimdFieldConfig> {
///         Some(SimdFieldConfig { modulus: GOLDILOCKS_P, element_bytes: 8 })
///     }
/// }
///
/// impl SimdRepr for MyGoldilocks {
///     fn modulus() -> u64 { GOLDILOCKS_P }
/// }
/// ```
///
/// # Example: Goldilocks cubic extension
///
/// ```ignore
/// #[derive(Clone, Copy, Debug, PartialEq,
///          zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
/// #[repr(transparent)]
/// struct MyExt3([u64; 3]);
///
/// impl SumcheckField for MyExt3 {
///     fn extension_degree() -> u64 { 3 }
///     fn _simd_field_config() -> Option<SimdFieldConfig> {
///         Some(SimdFieldConfig { modulus: GOLDILOCKS_P, element_bytes: 8 })
///     }
///     // ...
/// }
///
/// impl SimdRepr for MyExt3 {
///     fn modulus() -> u64 { GOLDILOCKS_P }
/// }
/// ```
pub trait SimdRepr:
    SumcheckField + zerocopy::IntoBytes + zerocopy::FromBytes + zerocopy::Immutable
{
    /// The base prime field modulus as a single `u64` limb.
    ///
    /// For the Goldilocks SIMD kernels to fire, this must equal
    /// [`GOLDILOCKS_P`] (`0xFFFF_FFFF_0000_0001`).
    fn modulus() -> u64;
}

/// SIMD field configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdFieldConfig {
    /// The base prime field modulus as a single `u64` limb.
    pub modulus: u64,
    /// Size of one base prime field element in bytes.
    pub element_bytes: usize,
}

/// Query SIMD configuration for a field type.
///
/// Returns `Some(config)` if the field's memory layout is SIMD-compatible.
///
/// Two paths to SIMD:
///
/// 1. **Arkworks types** (automatic): the `arkworks` feature detects
///    Goldilocks from the modulus at compile time. Zero overhead.
///
/// 2. **Non-arkworks types** (explicit): implement [`SimdRepr`] (requires
///    `zerocopy::IntoBytes + FromBytes + Immutable` — compiler-verified
///    layout) and override `_simd_field_config()`.
#[inline(always)]
pub fn simd_config<F: SumcheckField>() -> Option<SimdFieldConfig> {
    F::_simd_field_config()
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

        #[doc(hidden)]
        #[inline(always)]
        fn _to_raw_u64(self) -> u64 {
            debug_assert_eq!(core::mem::size_of::<Self>(), 8);
            // SAFETY: The caller has verified via `_simd_field_config()` that
            // Self is a Goldilocks-compatible field with `size_of::<Self>() == 8`
            // and `element_bytes == 8`. Arkworks `SmallFp` and `Fp<MontBackend<C, 1>, 1>`
            // both store exactly one u64 in Montgomery form as their sole field,
            // making this transmute a no-op reinterpretation.
            unsafe { core::mem::transmute_copy(&self) }
        }

        #[doc(hidden)]
        #[inline(always)]
        fn _from_raw_u64(raw: u64) -> Self {
            debug_assert_eq!(core::mem::size_of::<Self>(), 8);
            // SAFETY: Same layout guarantee as `_to_raw_u64` — the raw u64
            // is a valid Montgomery-form value for this Goldilocks field.
            unsafe { core::mem::transmute_copy(&raw) }
        }

        #[doc(hidden)]
        #[inline(always)]
        fn _as_u64_slice(slice: &[Self]) -> &[u64] {
            let d = <Self as ark_ff::Field>::extension_degree() as usize;
            debug_assert_eq!(core::mem::size_of::<Self>(), d * 8);
            let n_u64 = slice.len() * d;
            // SAFETY: Each element is `d` consecutive u64 values in Montgomery
            // form (verified by `_simd_field_config()`). The slice is contiguous
            // in memory, so the reinterpretation as `&[u64]` of length
            // `slice.len() * d` is valid. Lifetime and alignment are preserved
            // because u64 alignment (8) divides the field element alignment.
            unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u64, n_u64) }
        }

        #[doc(hidden)]
        #[inline(always)]
        fn _as_u64_slice_mut(slice: &mut [Self]) -> &mut [u64] {
            let d = <Self as ark_ff::Field>::extension_degree() as usize;
            debug_assert_eq!(core::mem::size_of::<Self>(), d * 8);
            let n_u64 = slice.len() * d;
            // SAFETY: Same layout guarantee as `_as_u64_slice`. Mutable access
            // is exclusive because we hold `&mut [Self]`.
            unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u64, n_u64) }
        }

        #[doc(hidden)]
        #[inline(always)]
        fn _from_u64_components(comps: &[u64]) -> Self {
            let d = <Self as ark_ff::Field>::extension_degree() as usize;
            assert_eq!(comps.len(), d);
            assert_eq!(core::mem::size_of::<Self>(), d * 8);
            // SAFETY: `comps` contains exactly `d` valid Montgomery-form u64
            // values and `size_of::<Self>() == d * 8` (verified by
            // `_simd_field_config()`). The copy reconstructs the in-memory
            // layout of the field element.
            unsafe {
                let mut out = core::mem::MaybeUninit::<Self>::uninit();
                core::ptr::copy_nonoverlapping(comps.as_ptr(), out.as_mut_ptr() as *mut u64, d);
                out.assume_init()
            }
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
