use ark_ff::Field;

use crate::{
    multilinear_product::{TimeProductProver, TimeProductProverConfig},
    prover::Prover,
    streams::Stream,
};

impl<F: Field, S: Stream<F>> Prover<F> for TimeProductProver<F, S> {
    type ProverConfig = TimeProductProverConfig<F, S>;
    type ProverMessage = Option<(F, F, F)>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        let num_variables = prover_config.num_variables;
        Self {
            claim: prover_config.claim,
            current_round: 0,
            evaluations: vec![None; prover_config.streams.len()],
            streams: Some(prover_config.streams),
            num_variables,
            inverse_four: F::from(4_u32).inverse().unwrap(),
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>) -> Option<(F, F, F)> {
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
            );
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
        multilinear_product::TimeProductProver,
        tests::{multilinear_product::consistency_test, BenchStream, F64},
    };

    #[test]
    fn parity_with_basic_prover() {
        consistency_test::<F64, BenchStream<F64>, TimeProductProver<F64, BenchStream<F64>>>();
    }
}
