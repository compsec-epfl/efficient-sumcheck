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

pub use inner_product_sumcheck::{
    batched_constraint_poly, inner_product_sumcheck, ProductSumcheck,
};
pub use multilinear_sumcheck::{multilinear_sumcheck, Sumcheck};

// ─── Internal / Advanced ─────────────────────────────────────────────────────

pub mod multilinear;
pub mod multilinear_product;
pub mod prover;
pub mod streams;

pub mod hypercube;
pub mod interpolation;
pub mod messages;
pub mod order_strategy;

#[doc(hidden)]
pub mod tests;
