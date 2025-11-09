use ark_ff::Field;
use ark_std::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    multilinear::reductions::variablewise,
    multilinear_product::provers::time::reductions::variablewise::{
        variablewise_product_evaluate, variablewise_product_evaluate_from_stream,
    },
    streams::Stream,
};

pub struct TimeProductProver<F: Field, S: Stream<F>> {
    pub current_round: usize,
    pub evaluations: Vec<Option<Vec<F>>>,
    pub streams: Option<Vec<S>>,
    pub num_variables: usize,
    pub inverse_four: F,
}

impl<F: Field, S: Stream<F>> TimeProductProver<F, S> {
    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
    pub fn num_free_variables(&self) -> usize {
        self.num_variables - self.current_round
    }
    /*
     * Note in evaluate() there's an optimization for the first round where we read directly
     * from the streams (instead of the tables), which reduces max memory usage by 1/2
     */
    pub fn vsbw_evaluate(&self) -> (F, F, F) {
        match &self.evaluations[0] {
            None => variablewise_product_evaluate_from_stream(
                &self.streams.clone().unwrap(),
                self.inverse_four,
            ),
            Some(_evals) => {
                let evals: Vec<Vec<F>> = self
                    .evaluations
                    .iter()
                    .filter_map(|opt| opt.clone()) // keep only Some(&Vec<F>)
                    .collect();
                let evals_slice: &[Vec<F>] = &evals;
                variablewise_product_evaluate(evals_slice, self.inverse_four)
            }
        }
    }
    pub fn vsbw_reduce_evaluations(&mut self, verifier_message: F, verifier_message_hat: F) {
        match &self.evaluations[0] {
            None => {
                let len = self.streams.clone().unwrap().len();
                for i in 0..len {
                    self.evaluations[i] = Some(vec![]);
                    variablewise::reduce_evaluations_from_stream(
                        &self.streams.as_mut().unwrap()[i],
                        self.evaluations[i].as_mut().unwrap(),
                        verifier_message,
                        verifier_message_hat,
                    );
                }
            }
            Some(_a) => {
                for table in &mut self.evaluations {
                    variablewise::reduce_evaluations(
                        table.as_mut().unwrap(),
                        verifier_message,
                        verifier_message_hat,
                    );
                }
            }
        }
    }
}
