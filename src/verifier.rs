//! Sumcheck verifier (Thaler Proposition 4.1).
//!
//! [`sumcheck_verify()`] checks a sumcheck proof against a claimed sum
//! and enforces the final oracle check via a caller-supplied closure.
//!
//! The closure-based design prevents a common footgun: forgetting to
//! verify `final_claim == g(r)` after the round checks pass. The caller
//! *must* provide an `oracle_check` function — the verifier calls it
//! automatically after all rounds succeed.
//!
//! For standalone use, pass [`default_oracle_check`] — it compares the
//! final claim against the prover's `final_value` from the proof.
//! For composed protocols (WHIR, GKR), pass a custom closure that
//! performs the check via PCS opening or delegation.

extern crate alloc;
use crate::field::SumcheckField;
use crate::proof::SumcheckError;
use crate::transcript::VerifierTranscript;
use alloc::vec;
use alloc::vec::Vec;

/// Default oracle check: verifies `final_claim == proof.final_value`.
///
/// Use this for standalone sumcheck verification. For composed protocols
/// (WHIR, GKR) where the oracle check is a PCS opening or delegation,
/// pass a custom closure instead.
///
/// ```ignore
/// let challenges = sumcheck_verify(
///     sum, degree, rounds, &mut t, noop_hook_verify,
///     default_oracle_check(proof.final_value),
/// )?;
/// ```
pub fn default_oracle_check<F: SumcheckField>(
    final_value: F,
) -> impl FnOnce(F, &[F]) -> Result<(), SumcheckError> {
    move |final_claim, _challenges| {
        if final_claim == final_value {
            Ok(())
        } else {
            Err(SumcheckError::FinalEvaluation)
        }
    }
}

/// Verify a sum-check proof against a claimed sum.
///
/// For each round j:
/// 1. Reads `degree + 1` evaluations from the transcript.
/// 2. Checks `g_j(0) + g_j(1) == current_claim`.
/// 3. Invokes `hook(round, transcript)`.
/// 4. Reads the verifier challenge `r_j`.
/// 5. Updates `current_claim = g_j(r_j)` via Lagrange interpolation.
///
/// After all rounds, calls `oracle_check(final_claim, &challenges)`.
/// The caller uses this to verify `final_claim == g(r_1, ..., r_v)` —
/// by direct evaluation, polynomial commitment opening, or GKR delegation.
///
/// Returns the challenge vector on success.
pub fn sumcheck_verify<F: SumcheckField, T: VerifierTranscript<F>>(
    claimed_sum: F,
    expected_degree: usize,
    num_rounds: usize,
    transcript: &mut T,
    mut hook: impl FnMut(usize, &mut T) -> Result<(), SumcheckError>,
    oracle_check: impl FnOnce(F, &[F]) -> Result<(), SumcheckError>,
) -> Result<Vec<F>, SumcheckError> {
    let mut claim = claimed_sum;
    let mut challenges = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        // Receive round polynomial evaluations from the prover.
        let num_evals = expected_degree + 1;
        let mut evals = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            let v = transcript
                .receive()
                .map_err(|_| SumcheckError::TranscriptError { round })?;
            evals.push(v);
        }

        // Consistency check: g_j(0) + g_j(1) == claim.
        let sum_01 = evals[0] + evals[1];
        if sum_01 != claim {
            return Err(SumcheckError::ConsistencyCheck { round });
        }

        // Per-round hook (e.g., PoW verification for WHIR).
        hook(round, transcript)?;

        // Squeeze verifier challenge.
        let r = transcript.challenge();
        challenges.push(r);

        // Update claim: g_j(r_j) via Lagrange interpolation.
        claim = evaluate_from_evals(&evals, r);
    }

    // Final oracle check: caller verifies final_claim == g(r_1, ..., r_v).
    oracle_check(claim, &challenges)?;

    Ok(challenges)
}

/// Evaluate a univariate polynomial from its evaluations at `{0, 1, ..., d}`
/// at an arbitrary point `r`.
///
/// Uses Lagrange interpolation:
///   g(r) = Σ_i g(i) · Π_{j≠i} (r − j) / (i − j)
fn evaluate_from_evals<F: SumcheckField>(evals: &[F], r: F) -> F {
    let d = evals.len(); // degree + 1
    if d == 0 {
        return F::ZERO;
    }
    if d == 1 {
        return evals[0];
    }
    if d == 2 {
        // Linear: g(r) = g(0) + r·(g(1) − g(0)).
        return evals[0] + r * (evals[1] - evals[0]);
    }

    // General case: Lagrange interpolation at {0, 1, ..., d-1}.
    // Precompute (r − j) for all j.
    let r_minus: Vec<F> = (0..d).map(|j| r - F::from_u64(j as u64)).collect();

    // Precompute prefix/suffix products of (r − j).
    let mut prefix = vec![F::ONE; d + 1];
    for i in 0..d {
        prefix[i + 1] = prefix[i] * r_minus[i];
    }
    let mut suffix = vec![F::ONE; d + 1];
    for i in (0..d).rev() {
        suffix[i] = suffix[i + 1] * r_minus[i];
    }

    // Precompute 1 / (i − j) for all j ≠ i, accumulated as products.
    // barycentric_weight[i] = 1 / Π_{j≠i} (i − j) = 1 / (i! · (d-1-i)! · (-1)^{d-1-i})
    let mut result = F::ZERO;
    for i in 0..d {
        let numerator = prefix[i] * suffix[i + 1]; // Π_{j≠i} (r − j)
        let denom: F = barycentric_weight(i, d);
        if let Some(inv) = denom.inverse() {
            result += evals[i] * numerator * inv;
        }
    }
    result
}

/// Barycentric weight: Π_{j≠i, 0≤j<d} (i − j).
fn barycentric_weight<F: SumcheckField>(i: usize, d: usize) -> F {
    let mut w = F::ONE;
    for j in 0..d {
        if j != i {
            let diff = i as i64 - j as i64;
            if diff > 0 {
                w *= F::from_u64(diff as u64);
            } else {
                w *= -F::from_u64((-diff) as u64);
            }
        }
    }
    w
}
