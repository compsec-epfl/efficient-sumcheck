use ark_ff::Field;

use crate::{
    messages::VerifierMessages,
    multilinear_product::{SpaceProductProver, SpaceProductProverConfig},
    prover::Prover,
    streams::{Stream, StreamIterator},
    order_strategy::SignificantBitOrder,
};

impl<F: Field, S: Stream<F>> Prover<F> for SpaceProductProver<F, S> {
    type ProverConfig = SpaceProductProverConfig<F, S>;
    type ProverMessage = Option<(F, F, F)>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {

        let stream_iterators = prover_config
            .streams
            .iter()
            .cloned()
            .map(|s| StreamIterator::<F, S, SignificantBitOrder>::new(s))
            .collect();

        Self {
            claim: prover_config.claim,
            stream_iterators: stream_iterators,
            verifier_messages: VerifierMessages::new(&vec![]),
            current_round: 0,
            num_variables: prover_config.num_variables,
            inverse_four: F::from(4_u32).inverse().unwrap(),
        }
    }

    fn next_message(&mut self, verifier_message: Self::VerifierMessage) -> Self::ProverMessage {
        // Ensure the current round is within bounds
        if self.current_round >= self.num_variables {
            return None;
        }

        // If it's not the first round, add the verifier message to verifier_messages
        if self.current_round != 0 {
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
        }

        // evaluate using cty
        let sums: (F, F, F) = self.cty_evaluate();

        // don't forget to increment the round
        self.current_round += 1;

        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear_product::SpaceProductProver,
        streams::MemoryStream,
        tests::{multilinear_product::sanity_test, F19},
    };

    #[test]
    fn sumcheck() {
        sanity_test::<F19, MemoryStream<F19>, SpaceProductProver<F19, MemoryStream<F19>>>();
    }
}
