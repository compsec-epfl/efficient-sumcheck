use ark_ff::Field;

use crate::{
    hypercube::Hypercube,
    messages::VerifierMessages,
    multilinear::{BlendyProver, BlendyProverConfig},
    order_strategy::GraycodeOrder,
    prover::Prover,
    streams::Stream,
};

impl<'a, F, S> Prover<F> for BlendyProver<F, S>
where
    F: Field,
    S: Stream<F>,
{
    type ProverConfig = BlendyProverConfig<F, S>;
    type ProverMessage = Option<(F, F)>;
    type VerifierMessage = Option<F>;

    fn new(prover_config: Self::ProverConfig) -> Self {
        let stage_size: usize = prover_config.num_variables / prover_config.num_stages;
        Self {
            claimed_sum: prover_config.claim,
            current_round: 0,
            evaluation_stream: prover_config.stream,
            num_stages: prover_config.num_stages,
            num_variables: prover_config.num_variables,
            verifier_messages: VerifierMessages::new(&vec![]),
            sums: vec![F::ZERO; Hypercube::<GraycodeOrder>::stop_value(stage_size)],
            lag_polys: vec![F::ONE; Hypercube::<GraycodeOrder>::stop_value(stage_size)],
            lag_polys_update: vec![F::ONE; Hypercube::<GraycodeOrder>::stop_value(stage_size)],
            stage_size,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>) -> Option<(F, F)> {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if !self.is_initial_round() {
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
        }

        // at start of stage do some stuff
        if self.is_start_of_stage() {
            self.sum_update();
            self.update_prefix_sums();
        }

        // update lag_polys based on previous round
        self.update_lag_polys();

        let sums: (F, F) = self.compute_round(&self.sums);

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial sums
        Some(sums)
    }

    fn claim(&self) -> F {
        self.claimed_sum
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear::BlendyProver,
        streams::MemoryStream,
        tests::{multilinear::sanity_test, F19},
    };

    #[test]
    fn sumcheck() {
        sanity_test::<F19, MemoryStream<F19>, BlendyProver<F19, MemoryStream<F19>>>();
    }
}
