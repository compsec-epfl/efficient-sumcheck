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
        multilinear::{BlendyProver, BlendyProverConfig, TimeProver},
        prover::{Prover, ProverConfig},
        tests::{BenchStream, SmallF19, SmallF19Mont, F19},
    };

    #[test]
    fn algorithm_consistency() {
        // take an evaluation stream
        let evaluation_stream: BenchStream<F19> = BenchStream::new(20);
        let claim = evaluation_stream.claimed_sum;
        // initialize the provers
        let mut blendy_k3_prover = BlendyProver::<F19, BenchStream<F19>>::new(
            BlendyProverConfig::new(claim, 3, 20, evaluation_stream.clone()),
        );
        let mut time_prover = TimeProver::<F19, BenchStream<F19>>::new(<TimeProver<
            F19,
            BenchStream<F19>,
        > as Prover<F19>>::ProverConfig::default(
            claim,
            20,
            evaluation_stream,
        ));
        // run them and get the transcript
        let blendy_prover_transcript = Sumcheck::<F19>::prove::<
            BenchStream<F19>,
            BlendyProver<F19, BenchStream<F19>>,
        >(&mut blendy_k3_prover, &mut ark_std::test_rng());
        let time_prover_transcript = Sumcheck::<F19>::prove::<
            BenchStream<F19>,
            TimeProver<F19, BenchStream<F19>>,
        >(&mut time_prover, &mut ark_std::test_rng());
        // ensure the transcript is identical
        assert_eq!(
            time_prover_transcript.prover_messages,
            blendy_prover_transcript.prover_messages
        );
    }

    #[test]
    fn algorithm_consistency_small_fp() {
        // take an evaluation stream
        let evaluation_stream: BenchStream<SmallF19> = BenchStream::new(20);
        let claim = evaluation_stream.claimed_sum;
        // initialize the provers
        let mut blendy_k3_prover = BlendyProver::<SmallF19, BenchStream<SmallF19>>::new(
            BlendyProverConfig::new(claim, 3, 20, evaluation_stream.clone()),
        );
        let mut time_prover =
            TimeProver::<SmallF19, BenchStream<SmallF19>>::new(<TimeProver<
                SmallF19,
                BenchStream<SmallF19>,
            > as Prover<SmallF19>>::ProverConfig::default(
                claim, 20, evaluation_stream
            ));
        // run them and get the transcript
        let blendy_prover_transcript = Sumcheck::<SmallF19>::prove::<
            BenchStream<SmallF19>,
            BlendyProver<SmallF19, BenchStream<SmallF19>>,
        >(&mut blendy_k3_prover, &mut ark_std::test_rng());
        let time_prover_transcript = Sumcheck::<SmallF19>::prove::<
            BenchStream<SmallF19>,
            TimeProver<SmallF19, BenchStream<SmallF19>>,
        >(&mut time_prover, &mut ark_std::test_rng());
        // ensure the transcript is identical
        assert_eq!(
            time_prover_transcript.prover_messages,
            blendy_prover_transcript.prover_messages
        );
    }

    #[test]
    fn algorithm_consistency_small_fp_mont() {
        // take an evaluation stream
        let evaluation_stream: BenchStream<SmallF19Mont> = BenchStream::new(20);
        let claim = evaluation_stream.claimed_sum;
        // initialize the provers
        let mut blendy_k3_prover = BlendyProver::<SmallF19Mont, BenchStream<SmallF19Mont>>::new(
            BlendyProverConfig::new(claim, 3, 20, evaluation_stream.clone()),
        );
        let mut time_prover =
            TimeProver::<SmallF19Mont, BenchStream<SmallF19Mont>>::new(<TimeProver<
                SmallF19Mont,
                BenchStream<SmallF19Mont>,
            > as Prover<SmallF19Mont>>::ProverConfig::default(
                claim,
                20,
                evaluation_stream,
            ));
        // run them and get the transcript
        let blendy_prover_transcript = Sumcheck::<SmallF19Mont>::prove::<
            BenchStream<SmallF19Mont>,
            BlendyProver<SmallF19Mont, BenchStream<SmallF19Mont>>,
        >(&mut blendy_k3_prover, &mut ark_std::test_rng());
        let time_prover_transcript = Sumcheck::<SmallF19Mont>::prove::<
            BenchStream<SmallF19Mont>,
            TimeProver<SmallF19Mont, BenchStream<SmallF19Mont>>,
        >(&mut time_prover, &mut ark_std::test_rng());
        // ensure the transcript is identical
        assert_eq!(
            time_prover_transcript.prover_messages,
            blendy_prover_transcript.prover_messages
        );
    }
}
