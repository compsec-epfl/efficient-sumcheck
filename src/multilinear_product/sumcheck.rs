use ark_ff::Field;
use ark_std::{rand::Rng, vec::Vec};

use crate::{
    interpolation::LagrangePolynomial, order_strategy::GraycodeOrder, prover::Prover,
    streams::Stream,
};

#[derive(Debug, PartialEq)]
pub struct ProductSumcheck<F: Field> {
    pub prover_messages: Vec<(F, F, F)>,
    pub verifier_messages: Vec<F>,
    pub is_accepted: bool,
}

impl<F: Field> ProductSumcheck<F> {
    pub fn prove<S, P>(prover: &mut P, rng: &mut impl Rng) -> Self
    where
        S: Stream<F>,
        P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F, F)>>,
    {
        // Initialize vectors to store prover and verifier messages
        let mut prover_messages: Vec<(F, F, F)> = vec![];
        let mut verifier_messages: Vec<F> = vec![];
        let mut is_accepted = true;

        // Run the protocol
        let mut verifier_message: Option<F> = None;
        while let Some(message) = prover.next_message(verifier_message) {
            let round_sum = message.0 + message.1;
            let is_round_accepted = match verifier_message {
                // If first round, compare to claimed_sum
                None => round_sum == prover.claim(),
                Some(prev_verifier_message) => {
                    verifier_messages.push(prev_verifier_message);
                    let prev_prover_message = prover_messages.last().unwrap();
                    round_sum
                        == LagrangePolynomial::<F, GraycodeOrder>::evaluate_from_three_points(
                            prev_verifier_message,
                            *prev_prover_message,
                        )
                }
            };

            // Handle how to proceed
            prover_messages.push(message);
            if !is_round_accepted {
                is_accepted = false;
                break;
            }

            verifier_message = Some(F::rand(rng));
        }

        // Return a Sumcheck struct with the collected messages and acceptance status
        ProductSumcheck {
            prover_messages,
            verifier_messages,
            is_accepted,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear_product::{BlendyProductProver, TimeProductProver},
        order_strategy::GraycodeOrder,
        tests::{multilinear_product::consistency_test, BenchStream, F64},
    };

    #[test]
    fn algorithm_consistency() {
        consistency_test::<F64, BenchStream<F64>, TimeProductProver<F64, BenchStream<F64>>>();
        consistency_test::<
            F64,
            BenchStream<F64>,
            BlendyProductProver<F64, BenchStream<F64>, GraycodeOrder>,
        >();
    }
}
