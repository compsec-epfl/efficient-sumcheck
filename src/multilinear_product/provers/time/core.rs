use ark_ff::Field;
use ark_std::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::streams::Stream;

pub struct TimeProductProver<F: Field, S: Stream<F>> {
    pub claim: F,
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
                None => panic!("Both streams and evaluations cannot be None"),
            },
        };

        #[cfg(feature = "parallel")]
        {
            let p_evals = self.evaluations[0].as_deref();
            let q_evals = self.evaluations[1].as_deref();
            let streams = self.streams.as_ref();

            let (acc00, acc01, acc10, acc11) = (0..evaluations_len / 2)
                .into_par_iter()
                // each worker gets its own local (j00, j01, j10, j11)
                .fold(
                    || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
                    |(mut j00, mut j01, mut j10, mut j11), i| {
                        // Load p,q at bit = 0 and 1. We only branch on the “source” (vec vs stream).
                        let (p0, p1) = if let Some(pe) = p_evals {
                            (pe[i], pe[i | bitmask])
                        } else {
                            let s =
                                &streams.expect("Both streams and evaluations cannot be None")[0];
                            (s.evaluation(i), s.evaluation(i | bitmask))
                        };

                        let (q0, q1) = if let Some(qe) = q_evals {
                            (qe[i], qe[i | bitmask])
                        } else {
                            let s =
                                &streams.expect("Both streams and evaluations cannot be None")[1];
                            (s.evaluation(i), s.evaluation(i | bitmask))
                        };

                        // Directly accumulate the 2x2 contributions (no temp x/y tables needed)
                        j00 += p0 * q0; // (0,0)
                        j11 += p1 * q1; // (1,1)
                        j01 += p0 * q1; // (0,1)
                        j10 += p1 * q0; // (1,0)

                        (j00, j01, j10, j11)
                    },
                )
                // combine thread-local partials
                .reduce(
                    || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
                    |(a00, a01, a10, a11), (b00, b01, b10, b11)| {
                        (a00 + b00, a01 + b01, a10 + b10, a11 + b11)
                    },
                );
            j_prime_table.0 .0 += acc00;
            j_prime_table.0 .1 += acc01;
            j_prime_table.1 .0 += acc10;
            j_prime_table.1 .1 += acc11;
        }

        #[cfg(not(feature = "parallel"))]
        // Iterate through evaluations
        for i in 0..(evaluations_len / 2) {
            // these must be zeroed out
            let mut x_table: (F, F) = (F::ZERO, F::ZERO);
            let mut y_table: (F, F) = (F::ZERO, F::ZERO);

            // get all the values
            let p_zero = match &self.evaluations[0] {
                None => match &self.streams {
                    Some(streams) => streams[0].evaluation(i),
                    None => panic!("Both streams and evaluations cannot be None"),
                },
                Some(evaluations_p) => evaluations_p[i],
            };
            let q_zero = match &self.evaluations[1] {
                None => match &self.streams {
                    Some(streams) => streams[1].evaluation(i),
                    None => panic!("Both streams and evaluations cannot be None"),
                },
                Some(evaluations_q) => evaluations_q[i],
            };
            let p_one = match &self.evaluations[0] {
                None => match &self.streams {
                    Some(streams) => streams[0].evaluation(i | bitmask),
                    None => panic!("Both streams and evaluations cannot be None"),
                },
                Some(evaluations_p) => evaluations_p[i | bitmask],
            };
            let q_one = match &self.evaluations[1] {
                None => match &self.streams {
                    Some(streams) => streams[1].evaluation(i | bitmask),
                    None => panic!("Both streams and evaluations cannot be None"),
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
        sum_half *= self.inverse_four;

        (sum_0, sum_1, sum_half)
    }
    pub fn vsbw_reduce_evaluations(&mut self, verifier_message: F, verifier_message_hat: F) {
        for i in 0..self.evaluations.len() {
            // Clone or initialize the evaluations vector
            let mut evaluations = match &self.evaluations[i] {
                Some(evaluations) => evaluations.clone(),
                None => match &self.streams {
                    Some(streams) => vec![
                        F::ZERO;
                        2usize.pow(streams[i].num_variables().try_into().unwrap())
                            / 2
                    ],
                    None => panic!("Both streams and evaluations cannot be None"),
                },
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
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evaluations) => evaluations[i0],
                };
                let point_evaluation_i1 = match &self.evaluations[i] {
                    None => match &self.streams {
                        Some(streams) => streams[i].evaluation(i1),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evaluations) => evaluations[i1],
                };
                // Update the i0-th evaluation based on the reduction operation
                evaluations[i0] = point_evaluation_i0 * verifier_message_hat
                    + point_evaluation_i1 * verifier_message;
            }

            #[cfg(feature = "parallel")]
            let vm = verifier_message;
            #[cfg(feature = "parallel")]
            let vmh = verifier_message_hat;
            #[cfg(feature = "parallel")]
            match (&self.evaluations[i], &self.streams) {
                // Read from slice
                (Some(src), _) => {
                    evaluations.par_iter_mut().enumerate().for_each(
                        |(i0, out): (usize, &mut F)| {
                            let i1 = i0 | setbit;
                            let p0 = src[i0];
                            let p1 = src[i1];
                            *out = p0 * vmh + p1 * vm; // <- write through &mut
                        },
                    );
                }
                // Read from stream
                (None, Some(streams)) => {
                    let s = &streams[i];
                    evaluations.par_iter_mut().enumerate().for_each(
                        |(i0, out): (usize, &mut F)| {
                            let i1 = i0 | setbit;
                            let p0 = s.evaluation(i0);
                            let p1 = s.evaluation(i1);
                            *out = p0 * vmh + p1 * vm; // <- write through &mut
                        },
                    );
                }
                (None, None) => panic!("Both streams and evaluations cannot be None"),
            }

            // Truncate the evaluations vector to the correct length
            evaluations.truncate(evaluations_len);

            // Update the internal state with the new evaluations vector
            self.evaluations[i] = Some(evaluations.clone());
        }
    }
}
