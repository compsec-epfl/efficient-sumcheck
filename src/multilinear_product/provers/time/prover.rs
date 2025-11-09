use ark_ff::Field;

use crate::{
    multilinear::{
        reductions::{pairwise, variablewise},
        ReduceMode,
    },
    multilinear_product::{
        reductions::{pairwise as product_pairwise, variablewise as product_variablewise},
        TimeProductProver, TimeProductProverConfig,
    },
    prover::Prover,
    streams::Stream,
};

impl<F: Field, S: Stream<F>> TimeProductProver<F, S> {
    fn next_message_pairwise(&mut self, verifier_message: Option<F>) -> Option<(F, F, F)> {
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if self.current_round != 0 {
            if self.current_round > 1 {
                for table in &mut self.evaluations {
                    pairwise::reduce_evaluations(table, verifier_message.unwrap());
                }
            } else {
                for i in 0..self.streams.len() {
                    pairwise::reduce_evaluations_from_stream(
                        &self.streams[i],
                        self.evaluations[i].as_mut(),
                        verifier_message.unwrap(),
                    );
                }
            }
        }

        // evaluate using vsbw
        let sums: (F, F, F) = match self.current_round == 0 {
            true => {
                product_pairwise::product_evaluate_from_stream(&self.streams, self.inverse_four)
            }
            false => product_pairwise::product_evaluate(&self.evaluations, self.inverse_four),
        };

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial
        Some(sums)
    }
    fn next_message_variablewise(&mut self, verifier_message: Option<F>) -> Option<(F, F, F)> {
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if self.current_round != 0 {
            if self.current_round > 1 {
                for table in &mut self.evaluations {
                    variablewise::reduce_evaluations(
                        table,
                        verifier_message.unwrap(),
                        F::ONE - verifier_message.unwrap(),
                    );
                }
            } else {
                for i in 0..self.streams.len() {
                    variablewise::reduce_evaluations_from_stream(
                        &self.streams[i],
                        self.evaluations[i].as_mut(),
                        verifier_message.unwrap(),
                        F::ONE - verifier_message.unwrap(),
                    );
                }
            }
        }

        // evaluate using vsbw
        let sums: (F, F, F) = match self.current_round == 0 {
            true => {
                product_variablewise::product_evaluate_from_stream(&self.streams, self.inverse_four)
            }
            false => product_variablewise::product_evaluate(&self.evaluations, self.inverse_four),
        };

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial
        Some(sums)
    }
}

impl<F: Field, S: Stream<F>> Prover<F> for TimeProductProver<F, S> {
    type ProverConfig = TimeProductProverConfig<F, S>;
    type ProverMessage = Option<(F, F, F)>;
    type VerifierMessage = Option<F>;

    fn new(prover_config: Self::ProverConfig) -> Self {
        let num_variables = prover_config.num_variables;
        Self {
            current_round: 0,
            evaluations: vec![vec![]; prover_config.streams.len()],
            streams: prover_config.streams,
            num_variables,
            inverse_four: F::from(4_u32).inverse().unwrap(),
            reduce_mode: prover_config.reduce_mode,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>) -> Option<(F, F, F)> {
        match self.reduce_mode {
            ReduceMode::Pairwise => self.next_message_pairwise(verifier_message),
            ReduceMode::Variablewise => self.next_message_variablewise(verifier_message),
        }
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
