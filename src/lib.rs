//! # efficient-sumcheck
//!
//! Space-efficient implementations of the sumcheck protocol with Fiat-Shamir support.
//!
//! ## Quick Start
//!
//! For most use cases, you need just two functions and a transcript:
//!
//! ```text
//! use efficient_sumcheck::{multilinear_sumcheck, inner_product_sumcheck};
//! use efficient_sumcheck::transcript::{Transcript, SpongefishTranscript, SanityTranscript};
//! ```
//!
//! - [`multilinear_sumcheck()`] — standard multilinear sumcheck: `∑_x p(x)`
//! - [`inner_product_sumcheck()`] — inner product sumcheck: `∑_x f(x)·g(x)`
//!
//! Both accept any [`Transcript`] implementation — either
//! [`SpongefishTranscript`](transcript::SpongefishTranscript) for real Fiat-Shamir, or
//! [`SanityTranscript`](transcript::SanityTranscript) for testing with random challenges.
//!
//! ## Advanced Usage
//!
//! For custom prover implementations, streaming evaluation access,
//! or specialized reduction strategies, the internal modules expose the full
//! prover machinery: [`multilinear`], [`multilinear_product`], [`prover`], [`streams`].

// ─── Primary API ─────────────────────────────────────────────────────────────

/// Transcript trait and backends (Spongefish, Sanity).
pub mod transcript;

mod inner_product_sumcheck;
mod multilinear_sumcheck;
mod whir_sumcheck;

pub use inner_product_sumcheck::{
    accumulate_sparse_evaluations, batched_constraint_poly, inner_product_sumcheck,
    inner_product_sumcheck_partial_with_hook, inner_product_sumcheck_with_hook, ProductSumcheck,
};
pub use multilinear_sumcheck::{
    multilinear_sumcheck, multilinear_sumcheck_partial_with_hook, multilinear_sumcheck_with_hook,
    Sumcheck,
};
pub use whir_sumcheck::{
    whir_sumcheck, whir_sumcheck_fused, whir_sumcheck_fused_partial_with_hook,
    whir_sumcheck_fused_with_hook, whir_sumcheck_partial_with_hook, whir_sumcheck_verify,
    whir_sumcheck_verify_with_hook, whir_sumcheck_with_hook,
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

pub mod simd_fields;
pub mod simd_ops;
pub mod simd_sumcheck;

#[doc(hidden)]
pub mod tests;
