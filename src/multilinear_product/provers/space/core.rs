use ark_ff::Field;

use crate::{
    hypercube::Hypercube,
    interpolation::LagrangePolynomial,
    messages::VerifierMessages,
    order_strategy::MSBOrder,
    streams::{Stream, StreamIterator},
};

pub struct SpaceProductProver<F: Field, S: Stream<F>> {
    pub current_round: usize,
    pub stream_iterators: Vec<StreamIterator<F, S, MSBOrder>>,
    pub num_variables: usize,
    pub verifier_messages: VerifierMessages<F>,
    pub inverse_four: F,
}

impl<F: Field, S: Stream<F>> SpaceProductProver<F, S> {
    pub fn cty_evaluate(&mut self) -> (F, F) {
        let mut a: F = F::ZERO;
        let mut b: F = F::ZERO;

        self.stream_iterators
            .iter_mut()
            .for_each(|stream_it| stream_it.reset());

        for (_, _) in Hypercube::<MSBOrder>::new(self.num_variables - self.current_round - 1) {
            if self.current_round == 0 {
                let p0 = self.stream_iterators[0].next().unwrap();
                let p1 = self.stream_iterators[0].next().unwrap();
                let q0 = self.stream_iterators[1].next().unwrap();
                let q1 = self.stream_iterators[1].next().unwrap();
                a += p0 * q0;
                b += p0 * q1 + p1 * q0;
            } else {
                let mut partial_sum_p_0 = F::ZERO;
                let mut partial_sum_p_1 = F::ZERO;
                let mut partial_sum_q_0 = F::ZERO;
                let mut partial_sum_q_1 = F::ZERO;

                let mut sequential_lag_poly: LagrangePolynomial<F, MSBOrder> =
                    LagrangePolynomial::new(&self.verifier_messages);
                for (_, _) in Hypercube::<MSBOrder>::new(self.current_round) {
                    let lag_poly = sequential_lag_poly.next().unwrap();
                    partial_sum_p_0 += self.stream_iterators[0].next().unwrap() * lag_poly;
                    partial_sum_q_0 += self.stream_iterators[1].next().unwrap() * lag_poly;
                }

                let mut sequential_lag_poly: LagrangePolynomial<F, MSBOrder> =
                    LagrangePolynomial::new(&self.verifier_messages);
                for (_, _) in Hypercube::<MSBOrder>::new(self.current_round) {
                    let lag_poly = sequential_lag_poly.next().unwrap();
                    partial_sum_p_1 += self.stream_iterators[0].next().unwrap() * lag_poly;
                    partial_sum_q_1 += self.stream_iterators[1].next().unwrap() * lag_poly;
                }

                a += partial_sum_p_0 * partial_sum_q_0;
                b += partial_sum_p_0 * partial_sum_q_1 + partial_sum_p_1 * partial_sum_q_0;
            }
        }
        (a, b)
    }
}
