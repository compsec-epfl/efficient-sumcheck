//! Protocol runner for the sum-check protocol (Thaler Proposition 4.1).
//!
//! [`sumcheck()`] drives a [`SumcheckProver`] through `num_rounds` rounds,
//! writing round polynomials to the transcript, invoking a per-round hook,
//! and reading verifier challenges. It returns a [`SumcheckProof`] containing
//! the round polynomials, challenges, and the prover's final claimed value.
//!
//! Partial execution (`num_rounds < v`) supports composed protocols like
//! GKR (one sumcheck per layer) and WHIR (partial rounds interleaved with
//! commit/open).

extern crate alloc;
use crate::field::SumcheckField;
use crate::proof::SumcheckProof;
use crate::sumcheck_prover::SumcheckProver;
use crate::transcript::ProverTranscript;
use alloc::vec::Vec;

/// Run the sum-check protocol for `num_rounds` rounds.
///
/// `hook` is called each round after the prover message is written and
/// before the verifier challenge is read. Pass `|_, _| {}` when no hook
/// is needed.
///
/// On return the prover has been advanced through `num_rounds` folds.
/// If `num_rounds == v` (full execution), `proof.final_value` is the
/// prover's claimed evaluation at the random point. For partial execution,
/// the caller retains `prover` and can continue or inspect post-state.
pub fn sumcheck<F, T, H, P>(
    prover: &mut P,
    num_rounds: usize,
    transcript: &mut T,
    mut hook: H,
) -> SumcheckProof<F>
where
    F: SumcheckField,
    T: ProverTranscript<F>,
    H: FnMut(usize, &mut T),
    P: SumcheckProver<F>,
{
    let mut round_polys: Vec<Vec<F>> = Vec::with_capacity(num_rounds);
    let mut challenges: Vec<F> = Vec::with_capacity(num_rounds);
    let mut prev_challenge: Option<F> = None;

    for round in 0..num_rounds {
        let evals = prover.round(prev_challenge);

        // Send evaluations to transcript.
        for &v in &evals {
            transcript.send(v);
        }
        round_polys.push(evals);

        // Per-round hook (e.g., proof-of-work grinding for WHIR).
        hook(round, transcript);

        // Squeeze verifier challenge.
        let r = transcript.challenge();
        challenges.push(r);
        prev_challenge = Some(r);
    }

    // Apply the final challenge so final_value() is correct.
    if let Some(r) = prev_challenge {
        prover.finalize(r);
    }

    let final_value = prover.final_value();

    SumcheckProof {
        round_polys,
        challenges,
        final_value,
    }
}
