//! The [`SumcheckProver`] trait — the extension point for all sumcheck
//! prover strategies and polynomial shapes.
//!
//! Implementors define *how* the round polynomial is computed from the
//! prover's internal state. The protocol runner ([`crate::runner::sumcheck`])
//! calls [`round()`](SumcheckProver::round) once per round; the caller
//! retains ownership of the prover and can inspect post-state after sumcheck
//! completes.

extern crate alloc;
use crate::field::SumcheckField;
use alloc::vec::Vec;

/// Prover side of the sum-check protocol (Thaler Proposition 4.1).
///
/// # Lifecycle
///
/// ```text
/// evals_0 = round(None)         // round 0: compute g_0 from initial state
/// evals_1 = round(Some(r_0))    // round 1: fold with r_0, compute g_1
/// ...
/// evals_{v-1} = round(Some(r_{v-2}))  // round v-1: fold with r_{v-2}, compute g_{v-1}
/// finalize(r_{v-1})             // apply the last challenge
/// value = final_value()         // g(r_0, ..., r_{v-1})
/// ```
///
/// # Post-state
///
/// Since the prover is passed as `&mut P`, the caller retains ownership
/// after sumcheck completes and can query prover-specific accessors:
///
/// ```ignore
/// let proof = sumcheck(&mut prover, n, &mut t, |_, _| {});
/// let (f_eval, g_eval) = prover.final_evaluations(); // prover-specific
/// ```
pub trait SumcheckProver<F: SumcheckField> {
    /// Maximum degree of the round polynomial in the current variable.
    fn degree(&self) -> usize;

    /// Compute the round polynomial and advance state.
    ///
    /// Returns `d = degree()` values per round in the **EvalsInfty** wire
    /// format:
    ///
    /// - `d == 1`: `[g_j(0)]` — verifier derives `g_j(1) = claim - g_j(0)`.
    /// - `d >= 2`: `[g_j(0), g_j(∞), g_j(2), g_j(3), ..., g_j(d-1)]`
    ///   where `g_j(∞)` is the leading coefficient (coefficient of `x^d`).
    ///   The verifier derives `g_j(1) = claim - g_j(0)` from the consistency
    ///   constraint `g_j(0) + g_j(1) = claim`.
    ///
    /// One wire element per round is saved versus sending explicit
    /// evaluations at `{0, 1, ..., d}`. The leading-coefficient form is
    /// also typically the cheapest round-polynomial contribution to compute
    /// for product-structured summands (see BDDT25, ePrint 2025/1117).
    ///
    /// - `challenge = None`: round 0 — compute from initial state.
    /// - `challenge = Some(r)`: fold/update state with the previous round's
    ///   challenge `r`, then compute the next round polynomial.
    fn round(&mut self, challenge: Option<F>) -> Vec<F>;

    /// Apply the final verifier challenge.
    ///
    /// Called once after the last round, before [`final_value()`](Self::final_value).
    /// The prover folds its internal state with `last_challenge` so that
    /// `final_value()` can return `g(r_1, ..., r_v)`.
    fn finalize(&mut self, last_challenge: F);

    /// After all rounds and [`finalize()`](Self::finalize): the claimed
    /// value `g(r_1, ..., r_v)`.
    fn final_value(&self) -> F;
}
