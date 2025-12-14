use ark_ff::Field;

use crate::multilinear::provers::time::reductions::ReduceMode;
use crate::{
    multilinear::{
        provers::time::reductions::{pairwise, variablewise},
        TimeProver, TimeProverConfig,
    },
    prover::Prover,
    streams::Stream,
};

impl<F: Field, S: Stream<F>> TimeProver<F, S> {
    fn next_message_pairwise(&mut self, verifier_message: Option<F>) -> Option<(F, F)> {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if self.current_round != 0 {
            if self.current_round > 1 {
                pairwise::reduce_evaluations(
                    self.evaluations.as_mut().unwrap(),
                    verifier_message.unwrap(),
                );
            } else {
                self.evaluations = Some(vec![]);
                pairwise::reduce_evaluations_from_stream(
                    &self.evaluation_streams[0],
                    self.evaluations.as_mut().unwrap(),
                    verifier_message.unwrap(),
                );
            }
        }

        // evaluate using vsbw
        let sums = match &self.evaluations {
            None => pairwise::evaluate_from_stream(&self.evaluation_streams[0]),
            Some(evaluations) => pairwise::evaluate(evaluations),
        };

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial
        Some(sums)
    }
    fn next_message_variablewise(&mut self, verifier_message: Option<F>) -> Option<(F, F)> {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if self.current_round != 0 {
            if self.current_round > 1 {
                variablewise::reduce_evaluations(
                    self.evaluations.as_mut().unwrap(),
                    verifier_message.unwrap(),
                    F::ONE - verifier_message.unwrap(),
                );
            } else {
                self.evaluations = Some(vec![]);
                variablewise::reduce_evaluations_from_stream(
                    &self.evaluation_streams[0],
                    self.evaluations.as_mut().unwrap(),
                    verifier_message.unwrap(),
                    F::ONE - verifier_message.unwrap(),
                );
            }
        }

        // evaluate using vsbw
        let sums = match &self.evaluations {
            None => variablewise::evaluate_from_stream(&self.evaluation_streams[0]),
            Some(evaluations) => variablewise::evaluate(evaluations),
        };

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial
        Some(sums)
    }
}

impl<F: Field, S: Stream<F>> Prover<F> for TimeProver<F, S> {
    type ProverConfig = TimeProverConfig<F, S>;
    type ProverMessage = Option<(F, F)>;
    type VerifierMessage = Option<F>;

    fn new(prover_config: Self::ProverConfig) -> Self {
        Self {
            // claim: prover_config.claim,
            current_round: 0,
            evaluations: None,
            evaluation_streams: prover_config.streams,
            num_variables: prover_config.num_variables,
            reduce_mode: prover_config.reduce_mode,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>) -> Option<(F, F)> {
        match self.reduce_mode {
            ReduceMode::Pairwise => self.next_message_pairwise(verifier_message),
            ReduceMode::Variablewise => self.next_message_variablewise(verifier_message),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear::TimeProver,
        streams::MemoryStream,
        tests::{
            multilinear::{pairwise_sanity_test, sanity_test},
            F19,
        },
    };

    #[test]
    fn sanity_pairwise() {
        pairwise_sanity_test::<F19, MemoryStream<F19>, TimeProver<F19, MemoryStream<F19>>>();
    }

    #[test]
    fn sanity() {
        sanity_test::<F19, MemoryStream<F19>, TimeProver<F19, MemoryStream<F19>>>();
    }
}
