//! Sumcheck proof and error types.

use crate::field::SumcheckField;
use core::fmt;

/// Output of the sumcheck protocol (Thaler Proposition 4.1).
///
/// Contains the prover's round polynomials, the verifier's challenges,
/// and the prover's claimed final evaluation. The verifier reconstructs
/// consistency checks from this data; the final oracle check (verifying
/// `final_value == g(r_1, ..., r_v)`) is the caller's responsibility.
#[derive(Clone, Debug)]
pub struct SumcheckProof<F: SumcheckField> {
    /// Round polynomial evaluations: `round_polys[j]` contains
    /// `g_j(0), g_j(1), ..., g_j(degree)`.
    pub round_polys: Vec<Vec<F>>,

    /// Verifier challenges `r_1, ..., r_v`.
    pub challenges: Vec<F>,

    /// Prover's claimed value `g(r_1, ..., r_v)`.
    pub final_value: F,
}

/// Sumcheck verification error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SumcheckError {
    /// Round `j` consistency check failed: `g_j(0) + g_j(1) != claim`.
    ConsistencyCheck {
        round: usize,
        expected: String,
        got: String,
    },
    /// Round polynomial has wrong degree.
    DegreeMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    /// Final evaluation mismatch.
    FinalEvaluation { expected: String, got: String },
    /// Transcript error (e.g., malformed prover message).
    TranscriptError { round: usize, detail: String },
    /// Per-round hook failed (e.g., proof-of-work verification).
    HookError { round: usize, detail: String },
}

impl fmt::Display for SumcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SumcheckError::ConsistencyCheck {
                round,
                expected,
                got,
            } => write!(
                f,
                "round {round}: consistency check failed: expected {expected}, got {got}"
            ),
            SumcheckError::DegreeMismatch {
                round,
                expected,
                got,
            } => write!(
                f,
                "round {round}: degree mismatch: expected <= {expected}, got {got}"
            ),
            SumcheckError::FinalEvaluation { expected, got } => {
                write!(
                    f,
                    "final evaluation mismatch: expected {expected}, got {got}"
                )
            }
            SumcheckError::TranscriptError { round, detail } => {
                write!(f, "round {round}: transcript error: {detail}")
            }
            SumcheckError::HookError { round, detail } => {
                write!(f, "round {round}: hook error: {detail}")
            }
        }
    }
}
