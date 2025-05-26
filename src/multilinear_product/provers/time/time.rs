use ark_ff::Field;
use ark_std::vec::Vec;

use crate::streams::Stream;

pub struct TimeProductProver<F: Field, S: Stream<F>> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: Vec<Option<Vec<F>>>,
    pub streams: Option<Vec<S>>,
    pub num_variables: usize,
    pub inverse_four: F,
}

impl<'a, F: Field, S: Stream<F>> TimeProductProver<F, S> {
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
        // Initialize accumulators
        let mut sum_half = F::ZERO;
        let mut j_prime_table: ((F, F), (F, F)) = ((F::ZERO, F::ZERO), (F::ZERO, F::ZERO));

        // Calculate the bitmask for the number of free variables
        let bitmask: usize = 1 << (self.num_free_variables() - 1);

        // Determine the length of evaluations to iterate through
        let evaluations_len = match &self.evaluations[0] {
            Some(evaluations) => evaluations.len(),
            None => match &self.streams {
                Some(streams) => 2usize.pow(streams[0].num_variables() as u32),
                None => panic!(
                    "Both streams and evaluations cannot be None"
                ),
            },
        };

        // Iterate through evaluations
        for i in 0..(evaluations_len / 2) {
            // these must be zeroed out
            let mut x_table: (F, F) = (F::ZERO, F::ZERO);
            let mut y_table: (F, F) = (F::ZERO, F::ZERO);

            // get all the values
            let p_zero = match &self.evaluations[0] {
                None => match &self.streams {
                    Some(streams) => streams[0].evaluation(i),
                    None => panic!(
                        "Both streams and evaluations cannot be None"
                    ),
                },
                Some(evaluations_p) => evaluations_p[i],
            };
            let q_zero = match &self.evaluations[1] {
                None => match &self.streams {
                    Some(streams) => streams[1].evaluation(i),
                    None => panic!(
                        "Both streams and evaluations cannot be None"
                    ),
                },
                Some(evaluations_q) => evaluations_q[i],
            };
            let p_one = match &self.evaluations[0] {
                None => match &self.streams {
                    Some(streams) => streams[0].evaluation(i | bitmask),
                    None => panic!(
                        "Both streams and evaluations cannot be None"
                    ),
                },
                Some(evaluations_p) => evaluations_p[i | bitmask],
            };
            let q_one = match &self.evaluations[1] {
                None => match &self.streams {
                    Some(streams) => streams[1].evaluation(i | bitmask),
                    None => panic!(
                        "Both streams and evaluations cannot be None"
                    ),
                },
                Some(evaluations_q) => evaluations_q[i | bitmask],
            };

            // update tables
            x_table.0 += p_zero;
            y_table.0 += q_zero;
            y_table.1 += q_one;
            x_table.1 += p_one;

            // update j_prime
            j_prime_table.0 .0 = j_prime_table.0 .0 + x_table.0 * y_table.0;
            j_prime_table.1 .1 = j_prime_table.1 .1 + x_table.1 * y_table.1;
            j_prime_table.0 .1 = j_prime_table.0 .1 + x_table.0 * y_table.1;
            j_prime_table.1 .0 = j_prime_table.1 .0 + x_table.1 * y_table.0;
        }

        // update
        let sum_0 = j_prime_table.0 .0;
        let sum_1 = j_prime_table.1 .1;
        sum_half +=
            j_prime_table.0 .0 + j_prime_table.1 .1 + j_prime_table.0 .1 + j_prime_table.1 .0;
        sum_half = sum_half * self.inverse_four;

        (sum_0, sum_1, sum_half)
    }
    pub fn vsbw_reduce_evaluations(&mut self, verifier_message: F, verifier_message_hat: F) {
        for i in 0..self.evaluations.len() {
            // Clone or initialize the evaluations vector
            let mut evaluations = match &self.evaluations[i] {
                Some(evaluations) => evaluations.clone(),
                None => {
                    match &self.streams {
                        Some(streams) => vec![
                            F::ZERO;
                            2usize.pow(streams[i].num_variables().try_into().unwrap()) / 2
                        ],
                        None => panic!(
                            "Both streams and evaluations cannot be None"
                        ),
                    }
                }
            };

            // Determine the length of evaluations to iterate through
            let evaluations_len = match &self.evaluations[i] {
                Some(evaluations) => evaluations.len() / 2,
                None => evaluations.len(),
            };

            // Calculate what bit needs to be set to index the second half of the last round's evaluations
            let setbit: usize = 1 << self.num_free_variables();

            // Iterate through pairs of evaluations
            for i0 in 0..evaluations_len {
                let i1 = i0 | setbit;

                // Get point evaluations for indices i0 and i1
                let point_evaluation_i0 = match &self.evaluations[i] {
                    None => match &self.streams {
                        Some(streams) => streams[i].evaluation(i0),
                        None => panic!(
                            "Both streams and evaluations cannot be None"
                        ),
                    },
                    Some(evaluations) => evaluations[i0],
                };
                let point_evaluation_i1 = match &self.evaluations[i] {
                    None => match &self.streams {
                        Some(streams) => streams[i].evaluation(i1),
                        None => panic!(
                            "Both streams and evaluations cannot be None"
                        ),
                    },
                    Some(evaluations) => evaluations[i1],
                };
                // Update the i0-th evaluation based on the reduction operation
                evaluations[i0] = point_evaluation_i0 * verifier_message_hat
                    + point_evaluation_i1 * verifier_message;
            }

            // Truncate the evaluations vector to the correct length
            evaluations.truncate(evaluations_len);

            // Update the internal state with the new evaluations vector
            self.evaluations[i] = Some(evaluations.clone());
        }
    }
}
