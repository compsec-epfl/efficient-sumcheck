use ark_ff::Field;
use ark_std::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    multilinear::{reductions::variablewise, ReduceMode},
    multilinear_product::reductions::variablewise as product_variablewise,
    streams::Stream,
};

pub struct TimeProductProver<F: Field, S: Stream<F>> {
    pub current_round: usize,
    pub evaluations: Vec<Vec<F>>,
    pub streams: Vec<S>,
    pub num_variables: usize,
    pub inverse_four: F,
    pub reduce_mode: ReduceMode,
}

impl<F: Field, S: Stream<F>> TimeProductProver<F, S> {
    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
    pub fn num_free_variables(&self) -> usize {
        self.num_variables - self.current_round
    }

    pub fn vsbw_evaluate(&self) -> (F, F, F) {
        match self.current_round == 0 {
            true => {
                product_variablewise::product_evaluate_from_stream(&self.streams, self.inverse_four)
            }
            false => product_variablewise::product_evaluate(&self.evaluations, self.inverse_four),
        }
    }
    pub fn vsbw_reduce_evaluations(&mut self, verifier_message: F, verifier_message_hat: F) {
        if self.current_round > 1 {
            for table in &mut self.evaluations {
                variablewise::reduce_evaluations(table, verifier_message, verifier_message_hat);
            }
        } else {
            for i in 0..self.streams.len() {
                variablewise::reduce_evaluations_from_stream(
                    &self.streams[i],
                    self.evaluations[i].as_mut(),
                    verifier_message,
                    verifier_message_hat,
                );
            }
        }
    }
}
