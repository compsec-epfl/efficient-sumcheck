use crate::{order_strategy::OrderStrategy, streams::Stream};
use ark_ff::Field;

/*
 * It's totally reasonable to use this when the evaluations table
 * fits in memory (and yes, it's not so much a stream in this case)
 */

#[derive(Debug, Clone)]
pub struct MemoryStream<F: Field> {
    pub evaluations: Vec<F>,
}

pub fn reorder_vec<F: Field, O: OrderStrategy>(evaluations: Vec<F>) -> Vec<F> {
    // abort if length not a power of two
    assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
    let num_vars = evaluations.len().trailing_zeros() as usize;
    let mut order = O::new(num_vars);
    let mut evaluations_ordered = Vec::with_capacity(evaluations.len());
    for index in &mut order {
        evaluations_ordered.push(evaluations[index]);
    }
    evaluations_ordered
}

impl<F: Field> MemoryStream<F> {
    pub fn new(evaluations: Vec<F>) -> Self {
        // abort if length not a power of two
        assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
        // return the MemoryStream instance
        Self { evaluations }
    }
    pub fn new_from_lex<O: OrderStrategy>(evaluations: Vec<F>) -> Self {
        // abort if length not a power of two
        assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
        Self::new(reorder_vec::<F, O>(evaluations))
    }
}

impl<F: Field> Stream<F> for MemoryStream<F> {
    fn evaluation(&self, point: usize) -> F {
        self.evaluations[point]
    }
    fn num_variables(&self) -> usize {
        self.evaluations.len().ilog2() as usize
    }
}
