use crate::streams::Stream;
use ark_ff::Field;

/*
 * Think about it: for benches that measure both wall time and memory
 * we need a stream that will cause reasonable field ops, but at the
 * same time, use only constant memory (nearly zero).
 *
 * As a solution, we use this stream that returns the index of the point evaluated
 * as a field value.
 */

#[derive(Debug, Clone)]
pub struct BenchStream<F: Field> {
    pub num_variables: usize,
    pub claimed_sum: F,
}
impl<F: Field> BenchStream<F> {
    pub fn new(num_variables: usize) -> Self {
        let hypercube_len = 2usize.pow(num_variables.try_into().unwrap());
        let mut claimed_sum: F = F::ZERO;
        for i in 0..hypercube_len {
            claimed_sum += F::from(i as u64);
        }
        Self {
            num_variables,
            claimed_sum,
        }
    }
}

impl<F: Field> Default for BenchStream<F> {
    fn default() -> Self {
        Self {
            num_variables: 0,
            claimed_sum: F::ZERO,
        }
    }
}

impl<F: Field> Stream<F> for BenchStream<F> {
    fn evaluation(&self, point: usize) -> F {
        F::from(point as u64)
    }
    fn num_variables(&self) -> usize {
        self.num_variables
    }
}
