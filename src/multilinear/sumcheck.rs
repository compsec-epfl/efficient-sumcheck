use ark_ff::Field;
use ark_std::{rand::Rng, vec::Vec};

use crate::{prover::Prover, streams::Stream};

#[derive(Debug)]
pub struct Sumcheck<F: Field> {
    pub prover_messages: Vec<(F, F)>,
    pub verifier_messages: Vec<F>,
    pub is_accepted: bool,
}

impl<F: Field> Sumcheck<F> {
    pub fn prove<S, P>(prover: &mut P, rng: &mut impl Rng) -> Self
    where
        S: Stream<F>,
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
        multilinear::{BlendyProver, BlendyProverConfig, ReduceMode, TimeProver},
        prover::{Prover, ProverConfig},
        tests::{BenchStream, F19},
    };

    #[test]
    fn sanity() {
        const NUM_VARIABLES: usize = 20;

        // take an evaluation stream
        let evaluation_stream: BenchStream<F19> = BenchStream::new(NUM_VARIABLES);
        let claim = evaluation_stream.claimed_sum;

        // blendy
        let mut blendy_k3_prover = BlendyProver::<F19, BenchStream<F19>>::new(
            BlendyProverConfig::new(claim, 3, NUM_VARIABLES, evaluation_stream.clone()),
        );
        let blendy_prover_transcript = Sumcheck::<F19>::prove::<
            BenchStream<F19>,
            BlendyProver<F19, BenchStream<F19>>,
        >(&mut blendy_k3_prover, &mut ark_std::test_rng());

        // time_prover_variablewise
        let mut time_prover_variablewise = TimeProver::<F19, BenchStream<F19>>::new(<TimeProver<
            F19,
            BenchStream<F19>,
        > as Prover<F19>>::ProverConfig::new(
            claim,
            NUM_VARIABLES,
            evaluation_stream.clone(),
            ReduceMode::Variablewise,
        ));
        let time_prover_variablewise_transcript =
            Sumcheck::<F19>::prove::<BenchStream<F19>, TimeProver<F19, BenchStream<F19>>>(
                &mut time_prover_variablewise,
                &mut ark_std::test_rng(),
            );

        // ensure transcripts identical
        assert_eq!(
            time_prover_variablewise_transcript.prover_messages,
            blendy_prover_transcript.prover_messages
        );

        // time_prover_pairwise: this should pass but I have nothing to compare it with
        let mut time_prover_pairwise = TimeProver::<F19, BenchStream<F19>>::new(<TimeProver<
            F19,
            BenchStream<F19>,
        > as Prover<F19>>::ProverConfig::default(
            claim,
            NUM_VARIABLES,
            evaluation_stream,
        ));
        let time_prover_pairwise_transcript =
            Sumcheck::<F19>::prove::<BenchStream<F19>, TimeProver<F19, BenchStream<F19>>>(
                &mut time_prover_pairwise,
                &mut ark_std::test_rng(),
            );
        assert!(time_prover_pairwise_transcript.is_accepted);
    }
}
