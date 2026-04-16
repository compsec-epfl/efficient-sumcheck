//! # efficient-sumcheck
//!
//! Sumcheck protocol implementations with Fiat-Shamir support.
//!
//! ## Quick Start
//!
//! Two primary entry points, both operating on evaluation vectors over the
//! boolean hypercube with a half-split (MSB) layout and a fused
//! fold+compute kernel:
//!
//! ```text
//! use efficient_sumcheck::{multilinear_sumcheck, inner_product_sumcheck};
//! use efficient_sumcheck::transcript::{Transcript, SpongefishTranscript, SanityTranscript};
//! ```
//!
//! - [`multilinear_sumcheck()`] — `∑_x v(x)` over a multilinear polynomial.
//! - [`inner_product_sumcheck()`] — `∑_x f(x)·g(x)` for two multilinears.
//!
//! Both accept any [`Transcript`] implementation — either
//! [`SpongefishTranscript`](transcript::SpongefishTranscript) for real
//! Fiat-Shamir, or [`SanityTranscript`](transcript::SanityTranscript) for
//! testing with seeded random challenges.
//!
//! ## Layout note
//!
//! The half-split (MSB) layout folds the top-most remaining variable each
//! round — round 0 splits `v[0..L/2]` vs `v[L/2..L]`. This differs from the
//! pair-split (LSB) layout used in earlier versions of this crate; callers
//! migrating from the old interface must reorder inputs by bit-reversal.

// ─── Primary API ─────────────────────────────────────────────────────────────

/// Transcript trait and backends (Spongefish, Sanity).
pub mod transcript;

mod inner_product_sumcheck;
mod multilinear_sumcheck;

pub use inner_product_sumcheck::{
    inner_product_sumcheck, inner_product_sumcheck_partial_with_hook,
    inner_product_sumcheck_verify, inner_product_sumcheck_verify_with_hook,
    inner_product_sumcheck_with_hook, ProductSumcheck,
};
pub use multilinear_sumcheck::{
    multilinear_sumcheck, multilinear_sumcheck_partial_with_hook, multilinear_sumcheck_verify,
    multilinear_sumcheck_verify_with_hook, multilinear_sumcheck_with_hook, Sumcheck,
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
