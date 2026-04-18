//! # efficient-sumcheck
//!
//! Sumcheck protocol (Thaler Proposition 4.1) with SIMD acceleration.
//!
//! ## Quick Start
//!
//! ```text
//! use efficient_sumcheck::field::SumcheckField;
//! use efficient_sumcheck::sumcheck_prover::SumcheckProver;
//! use efficient_sumcheck::runner::sumcheck;
//! use efficient_sumcheck::verifier::sumcheck_verify;
//! use efficient_sumcheck::provers::multilinear::MultilinearProver;
//! use efficient_sumcheck::provers::inner_product::InnerProductProver;
//! use efficient_sumcheck::fold;
//! ```
//!
//! The library is generic over any type implementing [`SumcheckField`](field::SumcheckField).
//! A blanket implementation for arkworks `Field` types is provided when the
//! `arkworks` feature is enabled (default).
//!
//! ## Architecture
//!
//! - **One protocol** (`runner::sumcheck`) parameterized by a `SumcheckProver`.
//! - **One verifier** (`verifier::sumcheck_verify`) for any degree.
//! - **Concrete provers**: `MultilinearProver` (d=1), `InnerProductProver` (d=2).
//! - **One fold** (Lemma 4.3), SIMD-accelerated for Goldilocks.
//! - MSB (half-split) layout throughout.
//! - SIMD for Goldilocks (NEON on aarch64, AVX-512 IFMA on x86_64) is
//!   transparent — zero overhead on non-Goldilocks fields.

// ─── Generic field trait ─────────────────────────────────────────────────────

pub mod field;
pub mod proof;

// ─── New canonical API (Thaler §4.1) ────────────────────────────────────────

pub mod sumcheck_prover;
pub mod runner;
pub mod verifier;
pub mod fold;
pub mod provers;

// ─── Primary API (legacy, to be replaced by the above) ─────────────────────

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

pub mod streams;

pub mod hypercube;
pub(crate) mod reductions;

pub mod coefficient_sumcheck;
pub mod folding;
pub mod poly_ops;

// SIMD internals — not part of the public API. SIMD dispatch is transparent
// through `fold`, `multilinear_sumcheck`, `inner_product_sumcheck`, etc.
pub(crate) mod simd_fields;
pub(crate) mod simd_sumcheck;

#[doc(hidden)]
pub mod tests;
