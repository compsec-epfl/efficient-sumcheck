use ark_ff::Field;
use ark_std::cfg_into_iter;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::{
    multilinear::{
        provers::time::reductions::{
            evaluate, evaluate_from_stream, reduce_evaluations, reduce_evaluations_from_stream,
        },
        TimeProver, TimeProverConfig,
    },
    prover::Prover,
    streams::Stream,
};

fn combine_streams<F: Field, S: Stream<F>>(streams: &[S]) -> Vec<F> {
    let len = 1usize << streams[0].num_variables();
    cfg_into_iter!(0..len)
        .map(|i| {
            streams.iter()
                .map(|s| s.evaluation(i))
                .fold(F::zero(), |acc, val| acc + val)
        })
        .collect()
}

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
            evaluation_streams: prover_config.streams,
            num_variables: prover_config.num_variables,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>) -> Option<(F, F)> {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        // TODO
        if self.current_round == 0 { 
            self.evaluations = Some(combine_streams(&self.evaluation_streams));
        } else if self.current_round > 1 || self.evaluation_streams.len() > 1 {
            reduce_evaluations(
                self.evaluations.as_mut().unwrap(),
                verifier_message.unwrap(),
                F::ONE - verifier_message.unwrap(),
            );
        } else {
            self.evaluations = Some(vec![]);
            reduce_evaluations_from_stream(
                &self.evaluation_streams[0],
                self.evaluations.as_mut().unwrap(),
                verifier_message.unwrap(),
                F::ONE - verifier_message.unwrap(),
            );
        }

        // evaluate using vsbw
        let sums = match &self.evaluations {
            None => evaluate_from_stream(&self.evaluation_streams[0]),
            Some(evaluations) => evaluate(evaluations),
        };

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
