use ark_ff::Field;

use crate::{
    multilinear::{TimeProver, TimeProverConfig},
    prover::Prover,
    streams::Stream,
};

impl<F: Field, S: Stream<F>> Prover<F> for TimeProver<F, S> {
    type ProverConfig = TimeProverConfig<F, S>;
    type ProverMessage = Option<(F, F)>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        Self {
            claim: prover_config.claim,
            current_round: 0,
            evaluations: None,
            evaluation_stream: prover_config.stream,
            num_variables: prover_config.num_variables,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>) -> Option<(F, F)> {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        // If it's not the first round, reduce the evaluations table
        if self.current_round != 0 {
            // update the evaluations table by absorbing leftmost variable assigned to verifier_message
            self.vsbw_reduce_evaluations(
                verifier_message.unwrap(),
                F::ONE - verifier_message.unwrap(),
            )
        }

        // evaluate using vsbw
        let sums = self.vsbw_evaluate();

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial
        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear::TimeProver,
        streams::MemoryStream,
        tests::{multilinear::pairwise_sanity_test, F19},
    };

    #[test]
    fn sumcheck() {
        pairwise_sanity_test::<F19, MemoryStream<F19>, TimeProver<F19, MemoryStream<F19>>>();
    }
}
