use ark_ff::Field;
use ark_std::vec::Vec;

use crate::{
    hypercube::Hypercube,
    interpolation::LagrangePolynomial,
    messages::VerifierMessages,
    order_strategy::{GraycodeOrder, OrderStrategy},
    streams::{Stream, StreamIterator},
};

pub struct BlendyProductProver<F: Field, S: Stream<F>, O: OrderStrategy> {
    pub claim: F,
    pub current_round: usize,
    pub streams: Vec<S>,
    pub stream_iterators: Vec<StreamIterator<F, S, O>>,
    pub num_stages: usize,
    pub num_variables: usize,
    pub max_rounds_phase1: usize,
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
}

impl<F: Field, S: Stream<F>, O: OrderStrategy> BlendyProductProver<F, S, O> {
    pub fn is_initial_round(&self) -> bool {
        self.current_round == 0
    }

    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }

    pub fn init_round_vars(&mut self) {
        let n = self.num_variables;
        let l = self.max_rounds_phase1;
        let j = self.current_round + 1;
        let (j_prime, t) = if j <= self.last_round_phase1 {
            let j_prime = 1usize << j.ilog2();
            let t = std::cmp::min(j_prime, n + 1 - j_prime);
            (j_prime, t)
        } else {
            let j_prime = self.last_round_phase1 + 1 + l * ((j - self.last_round_phase1 - 1) / l);
            let t = std::cmp::min(l, n + 1 - j_prime);
            (j_prime, t)
        };
        self.prev_table_round_num = j_prime;
        self.prev_table_size = t;
    }

    pub fn compute_round(&self) -> (F, F, F) {
        let mut sum_0 = F::ZERO;
        let mut sum_1 = F::ZERO;
        let mut sum_half = F::ZERO;

        // if first round, then no table is computed, need to compute sums from the streams
        if self.is_initial_round() {
            for (x_index, _) in Hypercube::<GraycodeOrder>::new(self.num_variables - 1) {
                let evaluation_point_0 = 0 << (self.num_variables - 1) | x_index;
                let evaluation_point_1 = 1 << (self.num_variables - 1) | x_index;
                let p0 = self.streams[0].evaluation(evaluation_point_0);
                let q0 = self.streams[1].evaluation(evaluation_point_0);
                let p1 = self.streams[0].evaluation(evaluation_point_1);
                let q1 = self.streams[1].evaluation(evaluation_point_1);
                sum_0 += p0 * q0;
                sum_1 += p1 * q1;
                sum_half += (p0 + p1) * (q0 + q1);
            }
        } else {
            // things to help iterating
            let b_prime_num_vars = self.current_round + 1 - self.prev_table_round_num;
            let v_num_vars: usize =
                self.prev_table_size + self.prev_table_round_num - self.current_round - 2;
            let b_prime_index_left_shift = v_num_vars + 1;

            // Lag Poly
            let mut sequential_lag_poly: LagrangePolynomial<F> =
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
        }
        sum_half = sum_half * self.inverse_four;
        (sum_0, sum_1, sum_half)
    }

    pub fn compute_state(&mut self) {
        let j = self.current_round + 1;
        let p = if j <= self.last_round_phase1 {
            (1 << j.ilog2()) == j // j is a power of 2
        } else {
            (j - self.last_round_phase1 - 1) % self.max_rounds_phase1 == 0 // the number of rounds since the last phase 1 round is a multiple of max_rounds_phase1
        };

        if p && !self.is_initial_round() {
            let j_prime = self.prev_table_round_num;
            let t = self.prev_table_size;

            // zero out the table
            let table_len = Hypercube::<GraycodeOrder>::stop_value(t);
            self.j_prime_table = vec![vec![F::ZERO; table_len]; table_len];
            self.x_table = vec![F::ZERO; table_len];
            self.y_table = vec![F::ZERO; table_len];

            // basically, this needs to get "zeroed" out at the beginning of state computation
            self.verifier_messages_round_comp = VerifierMessages::new_from_self(
                &self.verifier_messages,
                j_prime - 1,
                self.verifier_messages.messages.len(),
            );

            // some stuff for iterating
            let b_num_vars: usize = self.num_variables + 1 - j_prime - t;
            let x_num_vars = j_prime - 1;
            let x_index_left_shift = t + b_num_vars;

            // reset the streams
            self.stream_iterators
                .iter_mut()
                .for_each(|stream_it| stream_it.reset());

            for (b_index, _) in Hypercube::<GraycodeOrder>::new(b_num_vars) {
                for (b_prime_index, _) in Hypercube::<GraycodeOrder>::new(t) {
                    self.x_table[b_prime_index] = F::ZERO;
                    self.y_table[b_prime_index] = F::ZERO;
                    // LagPoly
                    let mut sequential_lag_poly: LagrangePolynomial<F> =
                        LagrangePolynomial::new(&self.verifier_messages);
                    let partial_point = b_prime_index << b_num_vars | b_index;
                    for (x_index, _) in Hypercube::<GraycodeOrder>::new(x_num_vars) {
                        // I imagine it's this loop taking lots of runtime
                        let evaluation_point = x_index << x_index_left_shift | partial_point;
                        print!("{},", evaluation_point);
                        let lag_poly = sequential_lag_poly.next().unwrap();
                        self.x_table[b_prime_index] +=
                            lag_poly * self.streams[0].evaluation(evaluation_point);
                        // assert_eq!(
                        //     self.streams[0].evaluation(evaluation_point),
                        //     self.stream_iterators[0].next().unwrap()
                        // );
                        self.y_table[b_prime_index] +=
                            lag_poly * self.streams[1].evaluation(evaluation_point);
                    }
                }
                for (b_prime_index, _) in Hypercube::<GraycodeOrder>::new(t) {
                    for (b_prime_prime_index, _) in Hypercube::<GraycodeOrder>::new(t) {
                        self.j_prime_table[b_prime_index][b_prime_prime_index] +=
                            self.x_table[b_prime_index] * self.y_table[b_prime_prime_index];
                    }
                }
            }
        }
        println!("");
    }
}
