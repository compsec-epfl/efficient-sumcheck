use ark_ff::Field;
use ark_std::cfg_iter_mut;
use ark_std::{cfg_into_iter, vec::Vec};

use crate::order_strategy::SignificantBitOrder;
use crate::{
    hypercube::Hypercube, interpolation::LagrangePolynomial, messages::VerifierMessages,
    order_strategy::GraycodeOrder, streams::Stream,
};
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

pub struct BlendyProver<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub current_round: usize,
    pub evaluation_stream: S,
    pub lag_polys: Vec<F>,
    pub lag_polys_update: Vec<F>,
    pub num_stages: usize,
    pub num_variables: usize,
    pub stage_size: usize,
    pub sums: Vec<F>,
    pub verifier_messages: VerifierMessages<F>,
}

impl<F, S> BlendyProver<F, S>
where
    F: Field,
    S: Stream<F>,
{
    fn shift_and_one_fill(num: usize, shift_amount: usize) -> usize {
        (num << shift_amount) | ((1 << shift_amount) - 1)
    }

    pub fn compute_round(&self, partial_sums: &[F]) -> (F, F) {
        // Initialize accumulators for sum_0 and sum_1
        let mut sum_0 = F::ZERO;
        let mut sum_1 = F::ZERO;

        // Calculate j_prime as j-(s-1)l
        let stage_start_index: usize = self.current_stage() * self.stage_size;
        let j_prime = self.current_round - stage_start_index;

        // Iterate through b2_start indices using Hypercube::new(j_prime + 1)
        for (b2_start_index, _) in Hypercube::<GraycodeOrder, SignificantBitOrder>::new(j_prime + 1)
        {
            // Calculate b2_start_index_0 and b2_start_index_1 for indexing partial_sums
            let shift_amount = if self.num_variables - stage_start_index < self.stage_size {
                // this is the oddly sized last stage when k doesn't divide num_vars
                self.num_variables - (self.current_stage() * self.stage_size) - j_prime - 1
            } else {
                self.stage_size - j_prime - 1
            };
            let b2_start_index_0: usize = b2_start_index << shift_amount;
            let b2_start_index_1: usize = Self::shift_and_one_fill(b2_start_index, shift_amount);

            // Calculate left_value and right_value based on partial_sums
            let left_value: F = match b2_start_index_0 {
                0 => F::ZERO,
                _ => partial_sums[b2_start_index_0 - 1],
            };
            let right_value = partial_sums[b2_start_index_1];
            let sum = right_value - left_value;

            // Match based on the last bit of b2_start
            match b2_start_index & 1 == 1 {
                false => sum_0 += self.lag_polys[b2_start_index] * sum,
                true => sum_1 += self.lag_polys[b2_start_index] * sum,
            }
        }

        // Return the accumulated sums
        (sum_0, sum_1)
    }

    fn current_stage(&self) -> usize {
        self.current_round / self.stage_size
    }

    pub fn is_initial_round(&self) -> bool {
        self.current_round == 0
    }

    pub fn is_start_of_stage(&self) -> bool {
        self.current_round.is_multiple_of(self.stage_size)
    }

    fn is_single_staged(&self) -> bool {
        self.num_stages == 1
    }

    pub fn sum_update(&mut self) {
        if self.is_single_staged() {
            return;
        };
        // 0. Declare ranges for convenience
        let b1_num_vars: usize = self.current_stage() * self.stage_size;
        let b2_num_vars: usize = if self.num_variables - b1_num_vars < self.stage_size {
            // this is the oddly sized last stage when k doesn't divide num_vars
            self.num_variables - b1_num_vars
        } else {
            self.stage_size
        };
        let b3_num_vars: usize = self.num_variables - b1_num_vars - b2_num_vars;

        // 1. Initialize SUM[b2] := 0 for each b2 ∈ {0,1}^l
        // we reuse self.sums we just have to zero out on the first access SEE BELOW

        // 2. Initialize st := LagInit((s - l)l, r)
        let mut sequential_lag_poly: LagrangePolynomial<F, GraycodeOrder> =
            LagrangePolynomial::new(&self.verifier_messages);

        // 3. For each b1 ∈ {0,1}^(s-1)l
        let len_sums: usize = self.sums.len();
        for (b1_index, _) in Hypercube::<GraycodeOrder, SignificantBitOrder>::new(b1_num_vars) {
            // (a) Compute (LagPoly, st) := LagNext(st)
            let lag_poly = sequential_lag_poly.next().unwrap();

            // (b) For each b2 ∈ {0,1}^l, for each b2 ∈ {0,1}^(k-s)l
            for (b2_index, _) in Hypercube::<GraycodeOrder, SignificantBitOrder>::new(b2_num_vars) {
                for (b3_index, _) in
                    Hypercube::<GraycodeOrder, SignificantBitOrder>::new(b3_num_vars)
                {
                    // Calculate the index for the current combination of b1, b2, and b3
                    let index = b1_index << (b2_num_vars + b3_num_vars)
                        | b2_index << b3_num_vars
                        | b3_index;

                    // Update SUM[b2]
                    self.sums[b2_index] =
                        match b1_index == 0 && b3_index == 0 && b2_index < len_sums {
                            // SEE HERE zero out the array on first access per update
                            true => lag_poly * self.evaluation_stream.evaluation(index),
                            false => {
                                self.sums[b2_index]
                                    + lag_poly * self.evaluation_stream.evaluation(index)
                            }
                        };
                }
            }
        }
    }
    pub fn update_lag_polys(&mut self) {
        // Calculate j_prime as j-(s-1)l
        let j_prime = self.current_round - (self.current_stage() * self.stage_size);

        // Iterate through b2_start indices using Hypercube::new(j_prime + 1)
        for (b2_start_index, _) in Hypercube::<GraycodeOrder, SignificantBitOrder>::new(j_prime + 1)
        {
            // calculate lag_poly from precomputed
            let lag_poly = match j_prime {
                0 => F::ONE,
                _ => {
                    let precomputed: F = *self.lag_polys.get(b2_start_index >> 1).unwrap();
                    match b2_start_index & 2 == 2 {
                        true => precomputed * *self.verifier_messages.messages.last().unwrap(),
                        false => precomputed * *self.verifier_messages.message_hats.last().unwrap(),
                    }
                }
            };
            self.lag_polys_update[b2_start_index] = lag_poly;
        }
        std::mem::swap(&mut self.lag_polys, &mut self.lag_polys_update);
    }

    pub fn update_prefix_sums(&mut self) {
        let n = self.sums.len();
        if n == 0 {
            return;
        }

        // Step 0: Unified input vector
        let input: Vec<F> = if self.is_single_staged() {
            (0..n)
                .map(|i| self.evaluation_stream.evaluation(i))
                .collect()
        } else {
            self.sums.clone()
        };

        // Step 1: Determine chunk size and boundaries
        #[cfg(feature = "parallel")]
        let num_threads = rayon::current_num_threads().max(1);
        #[cfg(not(feature = "parallel"))]
        let num_threads = 1;
        let chunk_size: usize = (n / num_threads).max(1);

        // Compute chunk start indices
        let chunk_starts: Vec<usize> = (0..n).step_by(chunk_size).collect();

        // Step 2: Parallel local prefix sums
        let mut partial_sums: Vec<Vec<F>> = cfg_into_iter!(chunk_starts.clone())
            .map(|start: usize| {
                let end: usize = (start + chunk_size).min(n);
                let mut local_sum = F::ZERO;
                let mut out = Vec::with_capacity(end - start);
                for &x in &input[start..end] {
                    local_sum += x;
                    out.push(local_sum);
                }
                out
            })
            .collect();

        // Step 3: Collect per-chunk totals
        let mut chunk_totals: Vec<F> = partial_sums
            .iter()
            .map(|chunk| *chunk.last().unwrap())
            .collect();

        // Step 4: Exclusive prefix sum of chunk totals (serial)
        for i in 1..chunk_totals.len() {
            let prev = chunk_totals[i - 1];
            chunk_totals[i] += prev;
        }

        // Step 5: Offset adjust each chunk in parallel
        cfg_iter_mut!(partial_sums)
            .enumerate()
            .for_each(|(i, chunk)| {
                let offset = if i == 0 { F::ZERO } else { chunk_totals[i - 1] };
                if offset != F::ZERO {
                    for x in chunk.iter_mut() {
                        *x += offset;
                    }
                }
            });

        // Step 6: Flatten into final result
        self.sums = partial_sums.into_iter().flatten().collect();
    }

    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
}
