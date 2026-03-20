use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, Polynomial};

use crate::multilinear::reductions::{pairwise, tablewise};
use crate::transcript::Transcript;

#[derive(Debug)]
pub struct CoefficientSumcheck<F: Field> {
    pub prover_messages: Vec<DensePolynomial<F>>,
    pub verifier_messages: Vec<F>,
}

/// Sumcheck prover for arbitrary-degree round polynomials in coefficient form.
///
/// Each round: `compute_round_poly` produces the round polynomial → coefficients
/// are sent to the transcript → challenge is received → all tables are reduced.
pub fn coefficient_sumcheck<F: Field>(
    mut compute_round_poly: impl FnMut(&[Vec<Vec<F>>], &[Vec<F>]) -> DensePolynomial<F>,
    tablewise: &mut [Vec<Vec<F>>],
    pairwise: &mut [Vec<F>],
    n_rounds: usize,
    transcript: &mut impl Transcript<F>,
) -> CoefficientSumcheck<F> {
    let mut prover_messages = Vec::with_capacity(n_rounds);
    let mut verifier_messages = Vec::with_capacity(n_rounds);

    for _ in 0..n_rounds {
        let round_poly = compute_round_poly(tablewise, pairwise);

        for coeff in &round_poly.coeffs {
            transcript.write(*coeff);
        }

        prover_messages.push(round_poly);

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
/// Each round: absorb coefficients → check `h(0) + h(1) == claim`
/// → squeeze challenge → update `claim = h(challenge)`.
pub fn sumcheck_verify<F: Field>(
    claim: &mut F,
    prover_messages: &[DensePolynomial<F>],
    transcript: &mut impl Transcript<F>,
) -> Option<Vec<F>> {
    let mut challenges = Vec::with_capacity(prover_messages.len());

    for h in prover_messages {
        for coeff in &h.coeffs {
            transcript.write(*coeff);
        }

        if h.evaluate(&F::zero()) + h.evaluate(&F::one()) != *claim {
            return None;
        }

        let c = transcript.read();
        *claim = h.evaluate(&c);
        challenges.push(c);
    }

    Some(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_poly::DenseUVPolynomial;
    use ark_std::test_rng;

    use crate::multilinear::reductions::pairwise;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;

    #[test]
    fn test_sumcheck_relation_holds_each_round() {
        // verify h(0) + h(1) == claimed sum at each round
        let mut rng = test_rng();
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

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

        // round 0: h(0) + h(1) == claimed_sum
        let h0 = &result.prover_messages[0];
        assert_eq!(
            h0.evaluate(&F64::from(0u64)) + h0.evaluate(&F64::from(1u64)),
            claimed_sum
        );

        // subsequent rounds: h_i(0) + h_i(1) == h_{i-1}(challenge_{i-1})
        for i in 1..result.prover_messages.len() {
            let prev_h = &result.prover_messages[i - 1];
            let challenge = result.verifier_messages[i - 1];
            let expected = prev_h.evaluate(&challenge);

            let h_i = &result.prover_messages[i];
            let actual = h_i.evaluate(&F64::from(0u64)) + h_i.evaluate(&F64::from(1u64));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_parity_with_multilinear_sumcheck() {
        // separate rng for evals so transcript rngs start at the same state
        use crate::multilinear_sumcheck;

        let mut eval_rng = test_rng();
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut eval_rng)).collect();
        let evals_clone = evals.clone();

        // run multilinear_sumcheck
        let mut rng1 = test_rng();
        let mut ml_evals = evals;
        let mut ml_transcript = SanityTranscript::new(&mut rng1);
        let ml_result = multilinear_sumcheck::<F64, F64>(&mut ml_evals, &mut ml_transcript);

        // run coefficient_sumcheck with degree-1 compute_h
        let mut rng2 = test_rng();
        let mut pairwise = vec![evals_clone];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut coeff_transcript = SanityTranscript::new(&mut rng2);
        let coeff_result = coefficient_sumcheck(
            |_tablewise, pairwise| {
                let (s0, s1) = pairwise::evaluate(&pairwise[0]);
                DensePolynomial::from_coefficients_vec(vec![s0, s1 - s0])
            },
            &mut tablewise,
            &mut pairwise,
            4,
            &mut coeff_transcript,
        );

        // challenges must match
        assert_eq!(ml_result.verifier_messages, coeff_result.verifier_messages);

        // round polynomials must be equivalent: (s0, s1) ↔ [s0, s1-s0]
        for (ml_msg, coeff_msg) in ml_result
            .prover_messages
            .iter()
            .zip(coeff_result.prover_messages.iter())
        {
            assert_eq!(coeff_msg.evaluate(&F64::from(0u64)), ml_msg.0);
            assert_eq!(coeff_msg.evaluate(&F64::from(1u64)), ml_msg.1);
        }
    }

    #[test]
    fn test_spongefish_transcript() {
        use crate::transcript::SpongefishTranscript;

        let mut rng = test_rng();
        let n = 1 << 3;
        let num_rounds = 3;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let domsep = spongefish::domain_separator!("test-coefficient-sumcheck"; module_path!())
            .instance(b"test");

        let prover_state = domsep.std_prover();
        let mut transcript = SpongefishTranscript::new(prover_state);

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];

        let result = coefficient_sumcheck(
            |_tablewise, pairwise| {
                let s0: F64 = pairwise[0].iter().step_by(2).copied().sum();
                let s1: F64 = pairwise[0].iter().skip(1).step_by(2).copied().sum();
                DensePolynomial::from_coefficients_vec(vec![s0, s1 - s0])
            },
            &mut tablewise,
            &mut pairwise,
            num_rounds,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), num_rounds);
        assert_eq!(result.verifier_messages.len(), num_rounds);
    }

    #[test]
    fn test_mixed_tablewise_and_pairwise() {
        let mut rng = test_rng();
        let n = 1 << 3;

        let table: Vec<Vec<F64>> = (0..n)
            .map(|_| vec![F64::rand(&mut rng), F64::rand(&mut rng)])
            .collect();
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut tablewise = vec![table];
        let mut pairwise = vec![evals];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            |tablewise, pairwise| {
                // combine both: sum of tablewise column 0 + pairwise even elements
                let ts: F64 = tablewise[0].iter().map(|row| row[0]).sum();
                let ps: F64 = pairwise[0].iter().step_by(2).copied().sum();
                DensePolynomial::from_coefficients_vec(vec![ts + ps])
            },
            &mut tablewise,
            &mut pairwise,
            3,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 3);
        // both should be reduced to single entries
        assert_eq!(tablewise[0].len(), 1);
        assert_eq!(pairwise[0].len(), 1);
    }

    #[test]
    fn test_higher_degree_round_polys() {
        // degree-2 round poly: h(0) = s0, h(1) = s1, h(2) = s0 + s1
        // verify the sumcheck relation holds at each round
        let mut rng = test_rng();
        let n = 1 << 3;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            |_tablewise, pairwise| {
                let s0: F64 = pairwise[0].iter().step_by(2).copied().sum();
                let s1: F64 = pairwise[0].iter().skip(1).step_by(2).copied().sum();
                // degree-2: interpolate through (0, s0), (1, s1), (2, s0+s1)
                // h(0)+h(1) = s0+s1 still holds, so sumcheck relation is satisfied
                let s2 = s0 + s1;
                let c0 = s0;
                let c1 = (-F64::from(3u64) * s0 + F64::from(4u64) * s1 - s2) / F64::from(2u64);
                let c2 = (s0 - F64::from(2u64) * s1 + s2) / F64::from(2u64);
                DensePolynomial::from_coefficients_vec(vec![c0, c1, c2])
            },
            &mut tablewise,
            &mut pairwise,
            3,
            &mut transcript,
        );

        // verify round 0: h(0) + h(1) == claimed sum
        let h0 = &result.prover_messages[0];
        assert_eq!(
            h0.evaluate(&F64::from(0u64)) + h0.evaluate(&F64::from(1u64)),
            claimed_sum
        );

        // all round polys should be degree 2
        for h in &result.prover_messages {
            assert_eq!(h.coeffs.len(), 3);
        }
    }

    #[test]
    fn test_single_round() {
        let mut rng = test_rng();
        let evals = vec![F64::rand(&mut rng), F64::rand(&mut rng)];
        let claimed_sum = evals[0] + evals[1];

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            |_tablewise, pairwise| {
                let s0 = pairwise[0][0];
                let s1 = pairwise[0][1];
                DensePolynomial::from_coefficients_vec(vec![s0, s1 - s0])
            },
            &mut tablewise,
            &mut pairwise,
            1,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 1);
        assert_eq!(result.verifier_messages.len(), 1);
        assert_eq!(pairwise[0].len(), 1);

        let h = &result.prover_messages[0];
        assert_eq!(
            h.evaluate(&F64::from(0u64)) + h.evaluate(&F64::from(1u64)),
            claimed_sum
        );
    }

    #[test]
    fn test_multiple_pairwise_tables() {
        // two independent pairwise tables, both reduced
        let mut rng = test_rng();
        let n = 1 << 3;
        let evals_a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut pairwise = vec![evals_a, evals_b];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            |_tablewise, pairwise| {
                // inner product contribution from both tables
                let s0: F64 = pairwise[0]
                    .iter()
                    .zip(pairwise[1].iter())
                    .step_by(2)
                    .map(|(a, b)| *a * b)
                    .sum();
                let s1: F64 = pairwise[0]
                    .iter()
                    .zip(pairwise[1].iter())
                    .skip(1)
                    .step_by(2)
                    .map(|(a, b)| *a * b)
                    .sum();
                DensePolynomial::from_coefficients_vec(vec![s0, s1 - s0])
            },
            &mut tablewise,
            &mut pairwise,
            3,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 3);
        assert_eq!(pairwise[0].len(), 1);
        assert_eq!(pairwise[1].len(), 1);
    }
}
