use ark_ff::Field;
use ark_std::vec::Vec;

#[cfg(feature = "parallel")]
use ark_std::cfg_into_iter;
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::streams::Stream;

pub struct TimeProver<F: Field, S: Stream<F>> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: Option<Vec<F>>,
    pub evaluation_stream: S, // Keep this for now, case we can do some small optimizations of first round etc
    pub num_variables: usize,
}

impl<F: Field, S: Stream<F>> TimeProver<F, S> {
    fn num_free_variables(&self) -> usize {
        self.num_variables - self.current_round
    }
    pub fn vsbw_evaluate(&self) -> (F, F) {
        // Calculate the bitmask for the number of free variables
        let bitmask: usize = 1 << (self.num_free_variables() - 1);

        // Determine the length of evaluations to iterate through
        let evaluations_len = match &self.evaluations {
            Some(evaluations) => evaluations.len(),
            None => 2usize.pow(self.evaluation_stream.num_variables() as u32),
        };

        #[cfg(feature = "parallel")]
        let (sum_0, sum_1) = cfg_into_iter!(0..evaluations_len)
            .map(|i| {
                // Get the point evaluation
                let val = if let Some(evals) = &self.evaluations {
                    evals[i]
                } else {
                    self.evaluation_stream.evaluation(i)
                };

                // Route value into the proper bucket
                if (i & bitmask) == 0 {
                    (val, F::zero()) // contributes to sum_0
                } else {
                    (F::zero(), val) // contributes to sum_1
                }
            })
            // Combine partial (sum0, sum1) pairs from each worker/thread.
            .reduce(
                || (F::zero(), F::zero()),
                |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
            );

        // Initialize accumulators for sum_0 and sum_1
        #[cfg(not(feature = "parallel"))]
        let mut sum_0 = F::ZERO;
        #[cfg(not(feature = "parallel"))]
        let mut sum_1 = F::ZERO;
        #[cfg(not(feature = "parallel"))]
        {
            // Iterate through evaluations
            for i in 0..evaluations_len {
                // Check if the bit at the position specified by the bitmask is set
                let is_set: bool = (i & bitmask) != 0;

                // Get the point evaluation for the current index
                let point_evaluation = match &self.evaluations {
                    Some(evaluations) => evaluations[i],
                    None => self.evaluation_stream.evaluation(i),
                };

                // Accumulate the value based on whether the bit is set or not
                match is_set {
                    false => sum_0 += point_evaluation,
                    true => sum_1 += point_evaluation,
                }
            }
        }

        // Return the accumulated sums
        (sum_0, sum_1)
    }

    pub fn vsbw_reduce_evaluations(&mut self, verifier_message: F, verifier_message_hat: F) {
        // Clone or initialize the evaluations vector
        #[cfg(feature = "parallel")]
        let is_first_go = self.evaluations.is_some();
        let mut evaluations = match &self.evaluations {
            Some(evaluations) => evaluations.clone(),
            None => vec![
                F::ZERO;
                2usize.pow(self.evaluation_stream.num_variables().try_into().unwrap()) / 2
            ],
        };

        // Determine the length of evaluations to iterate through
        let evaluations_len = match &self.evaluations {
            Some(evaluations) => evaluations.len() / 2,
            None => evaluations.len(),
        };

        // Calculate what bit needs to be set to index the second half of the last round's evaluations
        let setbit: usize = 1 << self.num_free_variables();

        #[cfg(feature = "parallel")]
        {
            // We'll write to the first half only.
            let dest = &mut evaluations[..evaluations_len];

            if is_first_go {
                // Read from the old immutable source (borrow, no extra clone).
                let src = self.evaluations.as_ref().unwrap();
                dest.par_iter_mut()
                    .enumerate()
                    .for_each(|(i0, slot): (usize, &mut F)| {
                        let i1 = i0 | setbit;
                        let v0 = src[i0];
                        let v1 = src[i1];
                        *slot = v0 * verifier_message_hat + v1 * verifier_message;
                    });
            } else {
                // Stream-only: compute both endpoints from the stream.
                let stream = &self.evaluation_stream;
                dest.par_iter_mut()
                    .enumerate()
                    .for_each(|(i0, slot): (usize, &mut F)| {
                        let i1 = i0 | setbit;
                        let v0 = stream.evaluation(i0);
                        let v1 = stream.evaluation(i1);
                        *slot = v0 * verifier_message_hat + v1 * verifier_message;
                    });
            }
        }

        // Iterate through pairs of evaluations
        #[cfg(not(feature = "parallel"))]
        for i0 in 0..evaluations_len {
            let i1 = i0 | setbit;

            // Get point evaluations for indices i0 and i1
            let point_evaluation_i0 = match &self.evaluations {
                None => self.evaluation_stream.evaluation(i0),
                Some(evaluations) => evaluations[i0],
            };
            let point_evaluation_i1 = match &self.evaluations {
                None => self.evaluation_stream.evaluation(i1),
                Some(evaluations) => evaluations[i1],
            };

            // Update the i0-th evaluation based on the reduction operation
            evaluations[i0] =
                point_evaluation_i0 * verifier_message_hat + point_evaluation_i1 * verifier_message;
        }

        // Truncate the evaluations vector to the correct length
        evaluations.truncate(evaluations_len);

        // Update the internal state with the new evaluations vector
        self.evaluations = Some(evaluations.clone());
    }
    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
}
