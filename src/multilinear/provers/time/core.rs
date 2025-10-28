use ark_ff::Field;
use ark_std::vec::Vec;

use crate::streams::Stream;

pub struct TimeProver<F: Field, S: Stream<F>> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: Option<Vec<F>>,
    pub evaluation_streams: Vec<S>, // TODO (z-tech): this can be released after the first call to vsbw_reduce_evaluations
    pub num_variables: usize,
}

impl<F: Field, S: Stream<F>> TimeProver<F, S> {
    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
}
