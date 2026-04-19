//! # effsc
//!
//! Sumcheck protocol (Thaler Proposition 4.1) with SIMD acceleration.

#![cfg_attr(not(feature = "arkworks"), no_std)]

extern crate alloc;

// ─── Generic field trait ─────────────────────────────────────────────────────

pub mod field;
pub mod proof;

// ─── New canonical API (Thaler §4.1) ────────────────────────────────────────

pub mod fold;
pub mod polynomial;
pub mod provers;
pub mod runner;
pub mod sumcheck_prover;
pub mod verifier;

/// No-op per-round hook for the prover. Pass to `sumcheck()` when no hook is needed.
///
/// ```ignore
/// let proof = sumcheck(&mut prover, n, &mut t, noop_hook);
/// ```
pub fn noop_hook<T>(_round: usize, _transcript: &mut T) {}

/// No-op per-round hook for the verifier. Pass to `sumcheck_verify()` when no hook is needed.
///
/// ```ignore
/// let result = sumcheck_verify(sum, deg, n, &mut t, noop_hook_verify)?;
/// ```
pub fn noop_hook_verify<T>(
    _round: usize,
    _transcript: &mut T,
) -> Result<(), crate::proof::SumcheckError> {
    Ok(())
}

// ─── Transcript ─────────────────────────────────────────────────────────────

pub mod transcript;

// ─── Arkworks-dependent modules ─────────────────────────────────────────────

#[cfg(feature = "arkworks")]
mod inner_product_sumcheck;
#[cfg(feature = "arkworks")]
mod multilinear_sumcheck;

#[cfg(feature = "arkworks")]
pub use inner_product_sumcheck::{
    inner_product_sumcheck, inner_product_sumcheck_partial, ProductSumcheck,
};
#[cfg(feature = "arkworks")]
pub use multilinear_sumcheck::{
    compute_sumcheck_polynomial, fold, fused_fold_and_compute_polynomial, multilinear_sumcheck,
    multilinear_sumcheck_partial, Sumcheck,
};

#[cfg(feature = "arkworks")]
pub mod coefficient_sumcheck;
#[cfg(feature = "arkworks")]
pub mod folding;
pub mod hypercube;
#[cfg(feature = "arkworks")]
pub mod poly_ops;
#[cfg(feature = "arkworks")]
pub(crate) mod reductions;
#[cfg(all(feature = "arkworks", feature = "simd"))]
pub(crate) mod simd_fields;
#[cfg(all(feature = "arkworks", feature = "simd"))]
pub(crate) mod simd_sumcheck;
#[cfg(feature = "arkworks")]
pub mod streams;
#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;
