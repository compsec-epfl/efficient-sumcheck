//! Sumcheck verifier (Thaler Proposition 4.1).
//!
//! [`sumcheck_verify()`] checks a sumcheck proof against a claimed sum
//! and returns the challenges along with the final claim.
//!
//! The verifier handles round checks (consistency, Lagrange interpolation).
//! The **oracle check** — verifying `final_claim == g(r)` — is the caller's
//! responsibility. This separation matches the textbook decomposition and
//! every real-world usage pattern:
//!
//! - **Standalone:** compare `result.final_claim == proof.final_value`
//! - **Composed (WHIR, GKR):** pass `result.final_claim` to the next layer
//! - **Custom (WARP):** compute the expected value from `result.challenges`
//!
//! # Wire format
//!
//! Round polynomials are communicated in the **EvalsInfty** format (see
//! [`crate::sumcheck_prover::SumcheckProver::round`]). The prover sends
//! `d = degree` values per round:
//!
//! - `d == 1`: `[h(0)]` — verifier derives `h(1) = claim - h(0)`.
//! - `d >= 2`: `[h(0), h(∞), h(2), ..., h(d-1)]`, where `h(∞)` is the
//!   leading coefficient. Verifier derives `h(1) = claim - h(0)` from the
//!   consistency constraint `h(0) + h(1) = claim`.

extern crate alloc;
use crate::field::SumcheckField;
use crate::proof::SumcheckError;
use crate::transcript::VerifierTranscript;
use alloc::vec;
use alloc::vec::Vec;

/// Output of [`sumcheck_verify`]: the verifier challenges and final claim.
///
/// The caller **must** verify `final_claim` — either by direct comparison,
/// PCS opening, or delegation to the next protocol layer.
#[derive(Clone, Debug)]
pub struct SumcheckResult<F: SumcheckField> {
    /// Verifier challenges `r_1, ..., r_v`.
    pub challenges: Vec<F>,
    /// The reduced claim after all rounds: `g_v(r_v)`.
    ///
    /// The caller must verify this equals `g(r_1, ..., r_v)` via an
    /// oracle query, polynomial commitment opening, or delegation.
    pub final_claim: F,
}

/// Verify a sum-check proof against a claimed sum.
///
/// For each round j:
/// 1. Reads `degree` values from the transcript (EvalsInfty wire format).
/// 2. Derives `h_j(1) = claim - h_j(0)` from the consistency constraint.
/// 3. Invokes `hook(round, transcript)`.
/// 4. Reads the verifier challenge `r_j`.
/// 5. Updates `claim = h_j(r_j)` via a polynomial reconstruction that
///    combines the given finite-point values with the leading coefficient.
///
/// Returns [`SumcheckResult`] containing the challenges and final claim.
/// The caller is responsible for the oracle check — verifying that
/// `final_claim == g(r_1, ..., r_v)`.
pub fn sumcheck_verify<F: SumcheckField, T: VerifierTranscript<F>>(
    claimed_sum: F,
    expected_degree: usize,
    num_rounds: usize,
    transcript: &mut T,
    mut hook: impl FnMut(usize, &mut T) -> Result<(), SumcheckError>,
) -> Result<SumcheckResult<F>, SumcheckError> {
    let mut claim = claimed_sum;
    let mut challenges = Vec::with_capacity(num_rounds);
    let d = expected_degree;

    for round in 0..num_rounds {
        // EvalsInfty wire format: receive `d` values per round (min 1).
        let n_wire = d.max(1);
        let mut recv = Vec::with_capacity(n_wire);
        for _ in 0..n_wire {
            let v = transcript
                .receive()
                .map_err(|_| SumcheckError::TranscriptError { round })?;
            recv.push(v);
        }

        let h0 = recv[0];
        let h1 = claim - h0;

        // Per-round hook (e.g., PoW verification for WHIR).
        hook(round, transcript)?;

        // Squeeze verifier challenge.
        let r = transcript.challenge();
        challenges.push(r);

        // Update claim: h_j(r_j).
        claim = if d == 0 {
            // Constant polynomial — claim stays equal to h0.
            h0
        } else {
            // h_inf = leading coefficient.
            // For d == 1: derive as h(1) - h(0) (slope).
            // For d >= 2: prover sends it explicitly as recv[1].
            let h_inf = if d >= 2 { recv[1] } else { h1 - h0 };

            // Build q-values at points {0, 1, ..., d-1}, where
            // q(x) = p(x) - h_inf * x^d has degree d-1.
            //   q(0) = h(0)
            //   q(1) = h(1) - h_inf
            //   q(i) = h(i) - h_inf * i^d   for i in 2..d  (d >= 3 only)
            let mut q_vals = Vec::with_capacity(d);
            q_vals.push(h0);
            if d >= 1 {
                q_vals.push(h1 - h_inf);
            }
            for i in 2..d {
                let hi = recv[i];
                let i_f = F::from_u64(i as u64);
                let mut i_d = F::ONE;
                for _ in 0..d {
                    i_d *= i_f;
                }
                q_vals.push(hi - h_inf * i_d);
            }

            // Degree-(d-1) Lagrange interpolation of q over {0, 1, ..., d-1}.
            let q_r = evaluate_from_evals(&q_vals, r);

            // p(r) = q(r) + h_inf * r^d
            let mut r_d = F::ONE;
            for _ in 0..d {
                r_d *= r;
            }
            q_r + h_inf * r_d
        };
    }

    Ok(SumcheckResult {
        challenges,
        final_claim: claim,
    })
}

/// Evaluate a univariate polynomial from its evaluations at `{0, 1, ..., d-1}`
/// at an arbitrary point `r`.
///
/// Uses Lagrange interpolation:
///   g(r) = Σ_i g(i) · Π_{j≠i} (r − j) / (i − j)
pub(crate) fn evaluate_from_evals<F: SumcheckField>(evals: &[F], r: F) -> F {
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
