//! Polynomial evaluation and arithmetic for sumcheck protocols.
//!
//! Generic over [`SumcheckField`](crate::field::SumcheckField) — available without the `arkworks` feature.
//!
//! # Evaluation
//!
//! - [`eval_horner`]: evaluate from coefficients at a point. O(d).
//! - [`eval_from_evals`]: evaluate from evaluations at `{0, 1, ..., d}` at
//!   an arbitrary point via barycentric Lagrange interpolation. O(d).
//! - [`BarycentricWeights`]: precompute weights once per degree, reuse
//!   across rounds for O(d) evaluation instead of O(d²).
//!
//! # Sequential Lagrange polynomial
//!
//! - [`SequentialLagrange`]: maintains `eq(r, x) = Π_j (r_j · x_j + (1-r_j)(1-x_j))`
//!   incrementally as you iterate over the hypercube. Composes with
//!   [`Ascending`](crate::hypercube::Ascending) for cache-friendly streaming.
//!
//! # Dense polynomial arithmetic
//!
//! Zero-allocation operations on coefficient slices:
//! - [`mul_into`], [`add_scaled`], [`eval_at`] (alias for `eval_horner`).

mod dense;
mod eval;
mod sequential_lagrange;

pub use dense::{add_scaled, eval_at, mul_into};
pub use eval::{eval_from_evals, eval_horner, BarycentricWeights};
pub use sequential_lagrange::SequentialLagrange;
