use ark_ff::Field;
use ark_std::{rand::Rng, vec::Vec};

use crate::{multilinear::Prover, streams::EvaluationStream};

#[derive(Debug)]
pub struct Sumcheck<F: Field> {
    pub prover_messages: Vec<(F, F)>,
    pub verifier_messages: Vec<F>,
    pub is_accepted: bool,
}

impl<F: Field> Sumcheck<F> {
    pub fn prove<S, P>(prover: &mut P, rng: &mut impl Rng) -> Self
    where
        S: EvaluationStream<F>,
        P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F)>>,
    {
        // Initialize vectors to store prover and verifier messages
        let mut prover_messages: Vec<(F, F)> = vec![];
        let mut verifier_messages: Vec<F> = vec![];
        let mut is_accepted = true;

        // Run the protocol
        let mut verifier_message: Option<F> = None;
        while let Some(message) = prover.next_message(verifier_message) {
            let round_sum = message.0 + message.1;
            let is_round_accepted = match verifier_message {
                // If first round, compare to claimed_sum
                None => round_sum == prover.claim(),
                // Else compute f(prev_verifier_msg) = prev_sum_0 - (prev_sum_0 - prev_sum_1) * prev_verifier_msg == round_sum, store verifier message
                Some(prev_verifier_message) => {
                    verifier_messages.push(prev_verifier_message);
                    let prev_prover_message = prover_messages.last().unwrap();
                    round_sum
                        == prev_prover_message.0
                            - (prev_prover_message.0 - prev_prover_message.1)
                                * prev_verifier_message
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
        Sumcheck {
            prover_messages,
            verifier_messages,
            is_accepted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Sumcheck;
    use crate::{
        multilinear::{BlendyProver, BlendyProverConfig, Prover, ProverConfig, TimeProver},
        tests::{BenchEvaluationStream, F19},
    };

    #[test]
    fn algorithm_consistency() {
        // take an evaluation stream
        let evaluation_stream: BenchEvaluationStream<F19> = BenchEvaluationStream::new(20);
        let claim = evaluation_stream.claimed_sum;
        // initialize the provers
        let mut blendy_k3_prover = BlendyProver::<F19, BenchEvaluationStream<F19>>::new(
            BlendyProverConfig::new(claim, 3, 20, evaluation_stream.clone()),
        );
        let mut time_prover =
            TimeProver::<F19, BenchEvaluationStream<F19>>::new(<TimeProver<
                F19,
                BenchEvaluationStream<F19>,
            > as Prover<F19>>::ProverConfig::default(
                claim, 20, evaluation_stream
            ));
        // run them and get the transcript
        let blendy_prover_transcript = Sumcheck::<F19>::prove::<
            BenchEvaluationStream<F19>,
            BlendyProver<F19, BenchEvaluationStream<F19>>,
        >(&mut blendy_k3_prover, &mut ark_std::test_rng());
        let time_prover_transcript = Sumcheck::<F19>::prove::<
            BenchEvaluationStream<F19>,
            TimeProver<F19, BenchEvaluationStream<F19>>,
        >(&mut time_prover, &mut ark_std::test_rng());
        // ensure the transcript is identical
        assert_eq!(
            time_prover_transcript.prover_messages,
            blendy_prover_transcript.prover_messages
        );
    }
}
