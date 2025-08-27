use ark_ff::Field;

use crate::{
    multilinear::{SpaceProver, SpaceProverConfig},
    prover::Prover,
    streams::Stream,
};

impl<F: Field, S: Stream<F>> Prover<F> for SpaceProver<F, S> {
    type ProverConfig = SpaceProverConfig<F, S>;
    type ProverMessage = Option<(F, F)>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        Self {
            claim: prover_config.claim,
            evaluation_stream: prover_config.stream,
            verifier_messages: Vec::<F>::with_capacity(prover_config.num_variables),
            verifier_message_hats: Vec::<F>::with_capacity(prover_config.num_variables),
            current_round: 0,
            num_variables: prover_config.num_variables,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>) -> Option<(F, F)> {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        // If it's not the first round, add the verifier message to verifier_messages
        if self.current_round != 0 {
            self.verifier_messages.push(verifier_message.unwrap());
            self.verifier_message_hats
                .push(F::ONE - verifier_message.unwrap());
        }

        // evaluate using cty
        let sums: (F, F) = self.cty_evaluate();

        // don't forget to increment the round
        self.current_round += 1;

        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear::SpaceProver,
        streams::MemoryStream,
        tests::{multilinear::sanity_test, SmallF19, F19},
    };

    #[test]
    fn sumcheck() {
        sanity_test::<F19, MemoryStream<F19>, SpaceProver<F19, MemoryStream<F19>>>();
    }

    #[test]
    fn sumcheck_small_fp() {
        sanity_test::<
            SmallF19,
            MemoryStream<SmallF19>,
            SpaceProver<SmallF19, MemoryStream<SmallF19>>,
        >();
    }
}
