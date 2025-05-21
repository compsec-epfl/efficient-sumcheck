use crate::{
    hypercube::Hypercube,
    interpolation::LagrangePolynomial,
    messages::VerifierMessages,
    multilinear_product::TimeProductProver,
    order_strategy::{GraycodeOrder, SignificantBitOrder},
    streams::{Stream, StreamIterator},
};
use ark_ff::Field;
use ark_std::vec::Vec;
use std::collections::BTreeSet;

pub struct BlendyProductProver<F: Field, S: Stream<F>> {
    pub claim: F,
    pub current_round: usize,
    pub streams: Vec<S>,
    pub stream_iterators: Vec<StreamIterator<F, S, SignificantBitOrder>>,
    pub num_stages: usize,
    pub num_variables: usize,
    pub last_round_phase1: usize,
    pub verifier_messages: VerifierMessages<F>,
    pub verifier_messages_round_comp: VerifierMessages<F>,
    pub x_table: Vec<F>,
    pub y_table: Vec<F>,
    pub j_prime_table: Vec<Vec<F>>,
    pub stage_size: usize,
    pub inverse_four: F,
    pub prev_table_round_num: usize,
    pub prev_table_size: usize,
    pub state_comp_set: BTreeSet<usize>,
    pub switched_to_vsbw: bool,
    pub vsbw_prover: TimeProductProver<F, S>,
}

impl<F: Field, S: Stream<F>> BlendyProductProver<F, S> {
    pub fn is_initial_round(&self) -> bool {
        self.current_round == 0
    }

    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }

    pub fn init_round_vars(&mut self) {
        let n = self.num_variables;
        let j = self.current_round + 1;

        if let Some(&prev_round) = self.state_comp_set.range(..=j).next_back() {
            self.prev_table_round_num = prev_round;
            if let Some(&next_round) = self.state_comp_set.range((j + 1)..).next() {
                self.prev_table_size = next_round - prev_round;
            } else {
                self.prev_table_size = n + 1 - prev_round;
            }
        } else {
            self.prev_table_round_num = 0;
            self.prev_table_size = 0;
        }
    }

    pub fn compute_round(&mut self) -> (F, F, F) {
        let mut sum_0 = F::ZERO;
        let mut sum_1 = F::ZERO;
        let mut sum_half = F::ZERO;

        // in the last rounds, we switch to the memory intensive prover
        if self.switched_to_vsbw {
            (sum_0, sum_1, sum_half) = self.vsbw_prover.vsbw_evaluate();
        }
        // if first few rounds, then no table is computed, need to compute sums from the streams
        else if self.current_round + 1 <= self.last_round_phase1 {
            // let time1 = std::time::Instant::now();

            // Lag Poly
            let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                LagrangePolynomial::new(&self.verifier_messages_round_comp);
            let lag_polys_len = Hypercube::<SignificantBitOrder>::stop_value(self.current_round);
            let mut lag_polys: Vec<F> = vec![F::ONE; lag_polys_len];

            // reset the streams
            self.stream_iterators
                .iter_mut()
                .for_each(|stream_it| stream_it.reset());

            for (x_index, _) in
                Hypercube::<SignificantBitOrder>::new(self.num_variables - self.current_round - 1)
            {
                // can avoid unnecessary additions for first round since there is no lag poly: gives a small speedup
                if self.is_initial_round() {
                    let p0 = self.stream_iterators[0].next().unwrap();
                    let p1 = self.stream_iterators[0].next().unwrap();
                    let q0 = self.stream_iterators[1].next().unwrap();
                    let q1 = self.stream_iterators[1].next().unwrap();
                    sum_0 += p0 * q0;
                    sum_1 += p1 * q1;
                    sum_half += (p0 + p1) * (q0 + q1);
                } else {
                    let mut partial_sum_p_0 = F::ZERO;
                    let mut partial_sum_p_1 = F::ZERO;
                    let mut partial_sum_q_0 = F::ZERO;
                    let mut partial_sum_q_1 = F::ZERO;
                    for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
                        if x_index == 0 {
                            lag_polys[b_index] = sequential_lag_poly.next().unwrap();
                        }
                        let lag_poly = lag_polys[b_index];
                        partial_sum_p_0 += self.stream_iterators[0].next().unwrap() * lag_poly;
                        partial_sum_q_0 += self.stream_iterators[1].next().unwrap() * lag_poly;
                    }
                    for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
                        let lag_poly = lag_polys[b_index];
                        partial_sum_p_1 += self.stream_iterators[0].next().unwrap() * lag_poly;
                        partial_sum_q_1 += self.stream_iterators[1].next().unwrap() * lag_poly;
                    }

                    sum_0 += partial_sum_p_0 * partial_sum_q_0;
                    sum_1 += partial_sum_p_1 * partial_sum_q_1;
                    sum_half +=
                        (partial_sum_p_0 + partial_sum_p_1) * (partial_sum_q_0 + partial_sum_q_1);
                }
            }
            sum_half = sum_half * self.inverse_four;
            // let time2 = std::time::Instant::now();
            // println!("round computation from stream took: {:?}", time2 - time1);
        }
        // computing evaluations from the cross product tables
        else {
            // things to help iterating
            let b_prime_num_vars = self.current_round + 1 - self.prev_table_round_num;
            let v_num_vars: usize =
                self.prev_table_size + self.prev_table_round_num - self.current_round - 2;
            let b_prime_index_left_shift = v_num_vars + 1;

            // Lag Poly
            let mut sequential_lag_poly: LagrangePolynomial<F, GraycodeOrder> =
                LagrangePolynomial::new(&self.verifier_messages_round_comp);
            let lag_polys_len = Hypercube::<GraycodeOrder>::stop_value(b_prime_num_vars);
            let mut lag_polys: Vec<F> = vec![F::ONE; lag_polys_len];

            // Sums
            for (b_prime_index, _) in Hypercube::<GraycodeOrder>::new(b_prime_num_vars) {
                for (b_prime_prime_index, _) in Hypercube::<GraycodeOrder>::new(b_prime_num_vars) {
                    // doing it like this, for each hypercube member lag_poly is computed exactly once
                    if b_prime_index == 0 {
                        lag_polys[b_prime_prime_index] = sequential_lag_poly.next().unwrap();
                    }

                    let lag_poly_1 = lag_polys[b_prime_index];
                    let lag_poly_2 = lag_polys[b_prime_prime_index];
                    let lag_poly = lag_poly_1 * lag_poly_2;
                    for (v_index, _) in Hypercube::<GraycodeOrder>::new(v_num_vars) {
                        let b_prime_0_v =
                            b_prime_index << b_prime_index_left_shift | 0 << v_num_vars | v_index;
                        let b_prime_prime_0_v = b_prime_prime_index << b_prime_index_left_shift
                            | 0 << v_num_vars
                            | v_index;
                        let b_prime_1_v =
                            b_prime_index << b_prime_index_left_shift | 1 << v_num_vars | v_index;
                        let b_prime_prime_1_v = b_prime_prime_index << b_prime_index_left_shift
                            | 1 << v_num_vars
                            | v_index;

                        sum_0 += lag_poly * self.j_prime_table[b_prime_0_v][b_prime_prime_0_v];
                        sum_1 += lag_poly * self.j_prime_table[b_prime_1_v][b_prime_prime_1_v];
                        sum_half += lag_poly
                            * (self.j_prime_table[b_prime_0_v][b_prime_prime_0_v]
                                + self.j_prime_table[b_prime_0_v][b_prime_prime_1_v]
                                + self.j_prime_table[b_prime_1_v][b_prime_prime_0_v]
                                + self.j_prime_table[b_prime_1_v][b_prime_prime_1_v]);
                    }
                }
            }
            sum_half = sum_half * self.inverse_four;
        }
        (sum_0, sum_1, sum_half)
    }

    pub fn compute_state(&mut self) {
        let j = self.current_round + 1;
        let p = self.state_comp_set.contains(&j);
        let is_largest = self.state_comp_set.range((j + 1)..).next().is_none();
        if p && !is_largest {
            // let time1 = std::time::Instant::now();
            let j_prime = self.prev_table_round_num;
            let t = self.prev_table_size;

            // println!(
            //     "table computation on round: {}, j_prime: {}, t: {}",
            //     j, j_prime, t
            // );

            // zero out the table
            let table_len = Hypercube::<SignificantBitOrder>::stop_value(t);
            self.j_prime_table = vec![vec![F::ZERO; table_len]; table_len];

            // basically, this needs to get "zeroed" out at the beginning of state computation
            self.verifier_messages_round_comp = VerifierMessages::new_from_self(
                &self.verifier_messages,
                j_prime - 1,
                self.verifier_messages.messages.len(),
            );

            // some stuff for iterating
            let b_num_vars: usize = self.num_variables + 1 - j_prime - t;
            let x_num_vars = j_prime - 1;

            // Lag Poly
            let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                LagrangePolynomial::new(&self.verifier_messages);

            assert!(x_num_vars == self.verifier_messages.messages.len());
            let lag_polys_len = Hypercube::<SignificantBitOrder>::stop_value(x_num_vars);
            let mut lag_polys: Vec<F> = vec![F::ONE; lag_polys_len];

            for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
                lag_polys[x_index] = sequential_lag_poly.next().unwrap();
            }

            // reset the streams
            self.stream_iterators
                .iter_mut()
                .for_each(|stream_it| stream_it.reset());

            // Ensure x_table and y_table are initialized with the correct size
            self.x_table = vec![F::ZERO; Hypercube::<SignificantBitOrder>::stop_value(t)];
            self.y_table = vec![F::ZERO; Hypercube::<SignificantBitOrder>::stop_value(t)];

            for (_, _) in Hypercube::<SignificantBitOrder>::new(b_num_vars) {
                for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(t) {
                    self.x_table[b_prime_index] = F::ZERO;
                    self.y_table[b_prime_index] = F::ZERO;

                    for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
                        self.x_table[b_prime_index] +=
                            lag_polys[x_index] * self.stream_iterators[0].next().unwrap();
                        self.y_table[b_prime_index] +=
                            lag_polys[x_index] * self.stream_iterators[1].next().unwrap();
                    }
                }
                for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(t) {
                    for (b_prime_prime_index, _) in Hypercube::<SignificantBitOrder>::new(t) {
                        self.j_prime_table[b_prime_index][b_prime_prime_index] +=
                            self.x_table[b_prime_index] * self.y_table[b_prime_prime_index];
                    }
                }
            }
            // let time2 = std::time::Instant::now();
            // println!("table computation took: {:?}", time2 - time1);
        } else if p && is_largest {
            // switch to the memory intensive sumcheck on the last round computation
            let num_variables_new = self.num_variables - j + 1;
            self.switched_to_vsbw = true;

            // println!(
            //     "switched to vsbw on round: {}, num_vars_new: {}",
            //     j, num_variables_new
            // );

            // reset the streams
            self.stream_iterators
                .iter_mut()
                .for_each(|stream_it| stream_it.reset());

            // initialize the evaluations for the memory-intensive implementation for the final rounds of the protocol
            let mut evaluations_p = vec![F::ZERO; 1 << num_variables_new];
            let mut evaluations_q = vec![F::ZERO; 1 << num_variables_new];

            for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(num_variables_new) {
                let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                    LagrangePolynomial::new(&self.verifier_messages);
                for (_, _) in Hypercube::<SignificantBitOrder>::new(j - 1) {
                    let lag_poly = sequential_lag_poly.next().unwrap();
                    evaluations_p[b_prime_index] +=
                        lag_poly * self.stream_iterators[0].next().unwrap();
                    evaluations_q[b_prime_index] +=
                        lag_poly * self.stream_iterators[1].next().unwrap();
                }
            }
            self.vsbw_prover.evaluations[0] = Some(evaluations_p);
            self.vsbw_prover.evaluations[1] = Some(evaluations_q);
        } else if self.switched_to_vsbw {
            let verifier_message = self.verifier_messages.messages[self.current_round - 1];
            self.vsbw_prover
                .vsbw_reduce_evaluations(verifier_message, F::ONE - verifier_message);
        }
    }
}
