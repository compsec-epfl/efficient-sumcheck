//! # efficient-sumcheck
//!
//! Sumcheck protocol implementations with Fiat-Shamir support.
//!
//! ## Quick Start
//!
//! ```text
//! use efficient_sumcheck::{multilinear_sumcheck, inner_product_sumcheck, fold};
//! use efficient_sumcheck::transcript::{Transcript, SpongefishTranscript, SanityTranscript};
//! ```
//!
//! - [`multilinear_sumcheck()`] — `∑_x v(x)` over a multilinear polynomial.
//! - [`inner_product_sumcheck()`] — `∑_x f(x)·g(x)` for two multilinears.
//! - [`fold()`] — MSB half-split fold, SIMD-accelerated for Goldilocks.
//!
//! Every entry point takes a per-round `hook: FnMut(round, &mut transcript)`
//! argument. Pass `|_, _| {}` when no hook is needed.
//!
//! ## Layout
//!
//! All operations use a half-split (MSB) layout: round `i` folds the
//! top-most remaining variable, splitting `v[0..L/2]` vs `v[L/2..L]`.
//! SIMD acceleration for Goldilocks (p = 2^64 − 2^32 + 1) is transparent —
//! no code changes needed. LLVM constant-folds the field detection at compile
//! time, so the non-SIMD path has zero overhead.

// ─── Primary API ─────────────────────────────────────────────────────────────

/// Transcript trait and backends (Spongefish, Sanity).
pub mod transcript;

mod inner_product_sumcheck;
mod multilinear_sumcheck;

pub use inner_product_sumcheck::{
    inner_product_sumcheck, inner_product_sumcheck_partial, inner_product_sumcheck_verify,
    ProductSumcheck,
};
pub use multilinear_sumcheck::{
    compute_sumcheck_polynomial, fold, fused_fold_and_compute_polynomial, multilinear_sumcheck,
    multilinear_sumcheck_partial, multilinear_sumcheck_verify, Sumcheck,
};

// ─── Internal / Advanced ─────────────────────────────────────────────────────

pub mod multilinear;
pub mod multilinear_product;
pub mod prover;
pub mod streams;

pub mod hypercube;
pub mod interpolation;
pub mod messages;
pub mod order_strategy;

pub mod coefficient_sumcheck;
pub mod folding;
pub mod poly_ops;

// SIMD internals — not part of the public API. SIMD dispatch is transparent
// through `fold`, `multilinear_sumcheck`, `inner_product_sumcheck`, etc.
pub(crate) mod simd_fields;
pub(crate) mod simd_sumcheck;

#[doc(hidden)]
pub mod tests;
