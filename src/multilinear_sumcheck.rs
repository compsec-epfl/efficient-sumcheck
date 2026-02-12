//! Standard multilinear sumcheck protocol.
//!
//! Given evaluations `[p(0..0), p(0..1), ..., p(1..1)]` of a multilinear polynomial `p`
//! on the boolean hypercube `{0,1}^n`, the [`multilinear_sumcheck`] function executes `n`
//! rounds of the sumcheck protocol and returns the resulting [`Sumcheck`] transcript.
//!
//! The function is parameterized by two field types:
//! - `BF` (base field): the field the evaluations live in
//! - `EF` (extension field): the field challenges are sampled from
//!
//! When no extension field is needed, set `EF = BF`.
//!
//! # Example
//!
//! ```text
//! use efficient_sumcheck::{multilinear_sumcheck, Sumcheck};
//! use efficient_sumcheck::transcript::SanityTranscript;
//!
//! // No extension field (BF = EF):
//! let mut evals = vec![F::from(1), F::from(2), F::from(3), F::from(4)];
//! let mut transcript = SanityTranscript::new(&mut rng);
//! let result: Sumcheck<F> = multilinear_sumcheck(&mut evals, &mut transcript);
//! ```

use ark_ff::Field;

use crate::multilinear::reductions::pairwise;
use crate::transcript::Transcript;

pub use crate::multilinear::Sumcheck;

/// Run the standard multilinear sumcheck protocol over an evaluation vector,
/// using a generic [`Transcript`] for Fiat-Shamir (or sanity/random challenges).
///
/// `BF` is the base field of the evaluations, `EF` is the extension field for challenges.
/// When `BF = EF`, this is the standard single-field sumcheck.
/// When `BF ≠ EF`, round 0 evaluates in `BF` and lifts to `EF`, then subsequent
/// rounds work entirely in `EF`.
///
/// Each round:
/// 1. Computes the round polynomial evaluations `(s(0), s(1))` via pairwise reduction.
/// 2. Writes them to the transcript (2 field elements).
/// 3. Reads the verifier's challenge from the transcript (1 field element).
/// 4. Reduces the evaluation vector by folding with the challenge.
pub fn multilinear_sumcheck<BF: Field, EF: Field + From<BF>>(
    evaluations: &mut Vec<BF>,
    transcript: &mut impl Transcript<EF>,
) -> Sumcheck<EF> {
    // checks
    assert!(
        evaluations.len().count_ones() == 1,
        "length must be a power of 2"
    );
    assert!(evaluations.len() >= 2, "need at least 1 variable");

    let num_rounds = evaluations.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = vec![];
    let mut verifier_messages: Vec<EF> = vec![];

    // ── Round 0: evaluate in BF, lift to EF, cross-field reduce ──
    if num_rounds > 0 {
        let msg_bf = pairwise::evaluate(evaluations);
        let msg = (EF::from(msg_bf.0), EF::from(msg_bf.1));

        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        let chg = transcript.read();
        verifier_messages.push(chg);

        // Cross-field reduce: BF evaluations + EF challenge → Vec<EF>
        let mut ef_evals = pairwise::cross_field_reduce(evaluations, chg);

        // Remaining rounds work in EF
        for _ in 1..num_rounds {
            let msg = pairwise::evaluate(&ef_evals);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            let chg = transcript.read();
            verifier_messages.push(chg);

            pairwise::reduce_evaluations(&mut ef_evals, chg);
        }
    }

    Sumcheck {
        verifier_messages,
        prover_messages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use crate::tests::F64;

    const NUM_VARS: usize = 4; // vectors of length 2^4 = 16

    #[test]
    fn test_multilinear_sumcheck_sanity() {
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut evaluations: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut transcript = SanityTranscript::new(&mut rng);
        let result = multilinear_sumcheck::<F64, F64>(&mut evaluations, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }

    #[test]
    fn test_multilinear_sumcheck_spongefish() {
        use crate::transcript::SpongefishTranscript;
        use spongefish::codecs::arkworks_algebra::FieldDomainSeparator;
        use spongefish::DomainSeparator;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut evaluations: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Build the IO pattern: each round absorbs 2 scalars and squeezes 1 challenge
        let mut domsep = DomainSeparator::new("test-multilinear-sumcheck");
        for _ in 0..NUM_VARS {
            domsep =
                <DomainSeparator as FieldDomainSeparator<F64>>::add_scalars(domsep, 2, "prover");
            domsep = <DomainSeparator as FieldDomainSeparator<F64>>::challenge_scalars(
                domsep, 1, "verifier",
            );
        }

        let prover_state = domsep.to_prover_state();
        let mut transcript = SpongefishTranscript::new(prover_state);
        let result = multilinear_sumcheck::<F64, F64>(&mut evaluations, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }
}
