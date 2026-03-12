use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
use spongefish::codecs::arkworks_algebra::{FieldToUnitDeserialize, UnitToField};
use spongefish::ProofError;

use crate::multilinear::reductions::{pairwise, tablewise};
use crate::transcript::Transcript;

#[derive(Debug)]
pub struct CoefficientSumcheck<F: Field> {
    pub prover_messages: Vec<DensePolynomial<F>>,
    pub verifier_messages: Vec<F>,
}

/// Sumcheck prover for arbitrary-degree round polynomials in coefficient form.
///
/// Each round: `compute_h` produces `h(X)` → coefficients are sent to the
/// transcript → challenge is received → all tables are reduced.
pub fn coefficient_sumcheck<F: Field>(
    mut compute_h: impl FnMut(&[Vec<Vec<F>>], &[Vec<F>]) -> DensePolynomial<F>,
    tablewise: &mut [Vec<Vec<F>>],
    pairwise: &mut [Vec<F>],
    n_rounds: usize,
    transcript: &mut impl Transcript<F>,
) -> CoefficientSumcheck<F> {
    let mut prover_messages = Vec::with_capacity(n_rounds);
    let mut verifier_messages = Vec::with_capacity(n_rounds);

    for _ in 0..n_rounds {
        let h = compute_h(tablewise, pairwise);

        for coeff in &h.coeffs {
            transcript.write(*coeff);
        }

        prover_messages.push(h);

        let c = transcript.read();
        verifier_messages.push(c);

        for table in tablewise.iter_mut() {
            tablewise::reduce_evaluations(table, c);
        }
        for table in pairwise.iter_mut() {
            pairwise::reduce_evaluations(table, c);
        }
    }

    CoefficientSumcheck {
        prover_messages,
        verifier_messages,
    }
}

/// Sumcheck verifier for arbitrary-degree round polynomials in coefficient form.
///
/// Each round: read `degree + 1` coefficients → check `h(0) + h(1) == target`
/// → update `target = h(challenge)`.
pub fn sumcheck_verify<F: Field>(
    degree: usize,
    target: &mut F,
    n_rounds: usize,
    verifier_state: &mut (impl FieldToUnitDeserialize<F> + UnitToField<F>),
) -> Result<Vec<F>, ProofError> {
    let mut challenges = Vec::with_capacity(n_rounds);

    for _ in 0..n_rounds {
        let mut h_coeffs = vec![F::zero(); degree + 1];
        verifier_state.fill_next_scalars(&mut h_coeffs)?;
        let h = DensePolynomial::from_coefficients_vec(h_coeffs);

        if h.evaluate(&F::zero()) + h.evaluate(&F::one()) != *target {
            return Err(ProofError::InvalidProof);
        }

        let [c] = verifier_state.challenge_scalars::<1>()?;
        *target = h.evaluate(&c);
        challenges.push(c);
    }

    Ok(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_poly::DenseUVPolynomial;
    use ark_std::test_rng;

    use crate::tests::F64;
    use crate::transcript::SanityTranscript;

    #[test]
    fn test_coefficient_sumcheck_pairwise_only() {
        // degree-1 round poly via pairwise evals, mimics multilinear_sumcheck
        let mut rng = test_rng();
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            |_tablewise, pairwise| {
                let s0: F64 = pairwise[0].iter().step_by(2).copied().sum();
                let s1: F64 = pairwise[0].iter().skip(1).step_by(2).copied().sum();
                DensePolynomial::from_coefficients_vec(vec![s0, s1 - s0])
            },
            &mut tablewise,
            &mut pairwise,
            4,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 4);
        assert_eq!(result.verifier_messages.len(), 4);
        assert_eq!(pairwise[0].len(), 1);
    }

    #[test]
    fn test_coefficient_sumcheck_tablewise_reduces() {
        let mut rng = test_rng();
        let n = 1 << 3;

        let table: Vec<Vec<F64>> = (0..n)
            .map(|_| vec![F64::rand(&mut rng), F64::rand(&mut rng)])
            .collect();
        let mut tablewise = vec![table];
        let mut pairwise: Vec<Vec<F64>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            |tablewise, _pairwise| {
                let s: F64 = tablewise[0].iter().map(|row| row[0]).sum();
                DensePolynomial::from_coefficients_vec(vec![s])
            },
            &mut tablewise,
            &mut pairwise,
            3,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 3);
        assert_eq!(result.verifier_messages.len(), 3);
        assert_eq!(tablewise[0].len(), 1);
    }

    #[test]
    fn test_coefficient_sumcheck_higher_degree() {
        // degree-3 round polynomial
        let mut rng = test_rng();
        let n = 1 << 2;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            |_tablewise, _pairwise| {
                DensePolynomial::from_coefficients_vec(vec![
                    F64::from(1u64),
                    F64::from(2u64),
                    F64::from(3u64),
                    F64::from(4u64),
                ])
            },
            &mut tablewise,
            &mut pairwise,
            2,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 2);
        assert_eq!(result.verifier_messages.len(), 2);
        assert_eq!(result.prover_messages[0].coeffs.len(), 4);
        assert_eq!(result.prover_messages[1].coeffs.len(), 4);
    }
}
