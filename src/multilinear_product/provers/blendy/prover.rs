use ark_ff::Field;

use crate::{
    messages::VerifierMessages,
    multilinear_product::{BlendyProductProver, BlendyProductProverConfig},
    prover::Prover,
    streams::{OrderStrategy, Stream, StreamIterator},
};

impl<F: Field, S: Stream<F>, O: OrderStrategy> Prover<F> for BlendyProductProver<F, S, O> {
    type ProverConfig = BlendyProductProverConfig<F, S>;
    type ProverMessage = Option<(F, F, F)>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        let num_variables: usize = prover_config.num_variables;
        let num_stages: usize = prover_config.num_stages;
        let stage_size: usize = num_variables / num_stages;
        let max_rounds_phase1: usize = num_variables.div_ceil(2 * num_stages);
        let stream_iterators = prover_config
            .streams
            .iter()
            .cloned()
            .map(|s| StreamIterator::new(s))
            .collect();
        // return the BlendyProver instance
        Self {
            claim: prover_config.claim,
            current_round: 0,
            streams: prover_config.streams,
            stream_iterators,
            num_stages,
            num_variables,
            max_rounds_phase1,
            last_round_phase1: (1usize << (num_variables.div_ceil(num_stages)).ilog2())
                + max_rounds_phase1
                - 1,
            verifier_messages: VerifierMessages::new(&vec![]),
            verifier_messages_round_comp: VerifierMessages::new(&vec![]),
            x_table: vec![],
            y_table: vec![],
            j_prime_table: vec![],
            stage_size,
            inverse_four: F::from(4_u32).inverse().unwrap(),
            prev_table_round_num: 0,
            prev_table_size: 0,
        }
    }

    fn next_message(&mut self, verifier_message: Self::VerifierMessage) -> Self::ProverMessage {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if !self.is_initial_round() {
            // this holds everything
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
            // this holds the randomness for between state computation r2
            self.verifier_messages_round_comp
                .receive_message(verifier_message.unwrap());
        }

        self.init_round_vars();

        self.compute_state();

        let sums: (F, F, F) = self.compute_round();

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial sums
        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear_product::BlendyProductProver,
        streams::GraycodeOrder,
        tests::{multilinear_product::consistency_test, BenchStream, F64},
    };

    #[test]
    fn parity_with_basic_prover() {
        consistency_test::<
            F64,
            BenchStream<F64>,
            BlendyProductProver<F64, BenchStream<F64>, GraycodeOrder>,
        >();
    }
}
