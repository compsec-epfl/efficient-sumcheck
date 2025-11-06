use ark_ff::Field;

use crate::{
    messages::VerifierMessages,
    prover::Prover,
    tests::multilinear::{BasicProver, BasicProverConfig},
};

impl<F: Field> Prover<F> for BasicProver<F> {
    type ProverConfig = BasicProverConfig<F>;
    type ProverMessage = Option<(F, F)>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        Self {
            claim: prover_config.claim,
            current_round: 0,
            num_variables: prover_config.num_variables,
            p: prover_config.p,
            verifier_messages: VerifierMessages::new(&vec![]),
        }
    }

    fn next_message(&mut self, verifier_message: Self::VerifierMessage) -> Self::ProverMessage {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if !self.is_initial_round() {
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
        }

        let sums: (F, F) = self.compute_round();

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial sums
        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use crate::prover::Prover;
    use crate::tests::polynomials::Polynomial;
    use crate::Sumcheck;
    use crate::{
        streams::{multivariate_claim, MemoryStream},
        tests::{
            multilinear::{sanity_test_driver, BasicProver, BasicProverConfig},
            polynomials::{three_variable_polynomial, three_variable_polynomial_evaluations},
            F19,
        },
    };
    use ark_poly::{
        multivariate::{self, SparsePolynomial, SparseTerm, Term},
        DenseMVPolynomial,
    };

    #[test]
    fn sanity() {
        let p: SparsePolynomial<F19, SparseTerm> = three_variable_polynomial();
        let s = MemoryStream::<F19>::new(three_variable_polynomial_evaluations());
        let config = BasicProverConfig::new(multivariate_claim(s.clone()), 3, p);
        let mut prover = BasicProver::new(config);
        sanity_test_driver(&mut prover);
    }

    #[test]
    fn sumcheck_1_variable() {
        // 3 * x_0 + 1
        let x_squared_plus_1 = multivariate::SparsePolynomial::from_coefficients_slice(
            1,
            &[
                (F19::from(3u32), multivariate::SparseTerm::new(vec![(0, 1)])),
                (F19::from(1u32), multivariate::SparseTerm::new(vec![])),
            ],
        );
        let s = MemoryStream::<F19>::new(x_squared_plus_1.to_evaluations());
        let config = BasicProverConfig::new(multivariate_claim(s.clone()), 1, x_squared_plus_1);
        let mut prover = BasicProver::new(config);
        let _transcript = Sumcheck::<F19>::prove::<MemoryStream<F19>, BasicProver<F19>>(
            &mut prover,
            &mut ark_std::test_rng(),
        );
        // println!("transcript: {:?}", transcript);
        // round 0
        // point: [0] -> 1
        // point: [1] -> 4
        // g0 = 3*x + 1 (it's the original polynomial so this is not useful for anything)
        // Sumcheck { prover_messages: [(1, 4)], verifier_messages: [], is_accepted: true }
    }

    #[test]
    fn sumcheck_2_variables() {
        // 3*x_0*x_1 + 5*x_0 + 1
        let x_zero_squared_x_one_plus_3_x_one_plus_1 =
            multivariate::SparsePolynomial::from_coefficients_slice(
                2,
                &[
                    (
                        F19::from(3u32),
                        multivariate::SparseTerm::new(vec![(0, 1), (1, 1)]),
                    ),
                    (F19::from(5u32), multivariate::SparseTerm::new(vec![(0, 1)])),
                    (F19::from(1u32), multivariate::SparseTerm::new(vec![])),
                ],
            );
        let s = MemoryStream::<F19>::new(x_zero_squared_x_one_plus_3_x_one_plus_1.to_evaluations());
        let config = BasicProverConfig::new(
            multivariate_claim(s.clone()),
            2,
            x_zero_squared_x_one_plus_3_x_one_plus_1,
        );
        let mut prover = BasicProver::new(config);
        let _transcript = Sumcheck::<F19>::prove::<MemoryStream<F19>, BasicProver<F19>>(
            &mut prover,
            &mut ark_std::test_rng(),
        );
        // println!("transcript: {:?}", transcript);

        // round 0
        // point: [0, 0] -> 1
        // point: [1, 0] -> 6
        // point: [0, 1] -> 1
        // point: [1, 1] -> 9
        // g0: 13*x + 2

        // round 1
        // point: [2, 0] -> 11
        // point: [2, 1] -> 17
        // g1: 6*x + 11

        // Sumcheck { prover_messages: [(2, 15), (11, 17)], verifier_messages: [2], is_accepted: true }
    }
}
