//! Sumcheck verifier (Thaler Proposition 4.1).
//!
//! [`sumcheck_verify()`] checks a sumcheck proof against a claimed sum.
//! It is generic over the polynomial degree — works for multilinear (d=1),
//! inner-product (d=2), or arbitrary-degree sumcheck.
//!
//! The verifier does NOT perform the final oracle check (Remark 4.2:
//! "the verifier can apply sumcheck even without knowing g"). The caller
//! is responsible for verifying `final_value == g(r_1, ..., r_v)` via
//! direct evaluation, delegation, or polynomial commitment.

use crate::field::SumcheckField;
use crate::proof::SumcheckError;
use crate::transcript::Transcript;

/// Verify a sum-check proof against a claimed sum.
///
/// For each round j:
/// 1. Reads `degree + 1` evaluations from the transcript.
/// 2. Checks `g_j(0) + g_j(1) == current_claim`.
/// 3. Invokes `hook(round, transcript)`.
/// 4. Reads the verifier challenge `r_j`.
/// 5. Updates `current_claim = g_j(r_j)` via Horner evaluation.
///
/// Returns `(final_claim, challenges)` on success. The caller checks
/// `final_claim == g(r_1, ..., r_v)`.
pub fn sumcheck_verify<F, T, H>(
    claimed_sum: F,
    expected_degree: usize,
    num_rounds: usize,
    transcript: &mut T,
    mut hook: H,
) -> Result<(F, Vec<F>), SumcheckError>
where
    F: SumcheckField,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    let mut claim = claimed_sum;
    let mut challenges = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        // Receive round polynomial evaluations from the prover.
        let num_evals = expected_degree + 1;
        let mut evals = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            let v = transcript
                .receive()
                .map_err(|e| SumcheckError::TranscriptError {
                    round,
                    detail: format!("{:?}", e),
                })?;
            evals.push(v);
        }

        // Consistency check: g_j(0) + g_j(1) == claim.
        let sum_01 = evals[0] + evals[1];
        if sum_01 != claim {
            return Err(SumcheckError::ConsistencyCheck {
                round,
                expected: format!("{:?}", claim),
                got: format!("{:?}", sum_01),
            });
        }

        // Per-round hook.
        hook(round, transcript);

        // Squeeze verifier challenge.
        let r = transcript.challenge();
        challenges.push(r);

        // Update claim: g_j(r_j) via Horner's method.
        // evals = [g(0), g(1), ..., g(d)]
        // We need to interpolate and evaluate at r.
        claim = evaluate_from_evals(&evals, r);
    }

    Ok((claim, challenges))
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
