use ark_ff::Field;

use crate::{
    hypercube::Hypercube,
    interpolation::LagrangePolynomial,
    messages::VerifierMessages,
    order_strategy::SignificantBitOrder,
    streams::{Stream, StreamIterator},
};

pub struct SpaceProductProver<F: Field, S: Stream<F>> {
    pub current_round: usize,
    pub stream_iterators: Vec<StreamIterator<F, S, SignificantBitOrder>>,
    pub num_variables: usize,
    pub verifier_messages: VerifierMessages<F>,
    pub inverse_four: F,
}

impl<F: Field, S: Stream<F>> SpaceProductProver<F, S> {
    pub fn cty_evaluate(&mut self) -> (F, F, F) {
        let mut sum_0: F = F::ZERO;
        let mut sum_1: F = F::ZERO;
        let mut sum_half: F = F::ZERO;

        // reset the streams
        self.stream_iterators
            .iter_mut()
            .for_each(|stream_it| stream_it.reset());

        for (_, _) in Hypercube::<SignificantBitOrder, SignificantBitOrder>::new(
            self.num_variables - self.current_round - 1,
        ) {
            // can avoid unnecessary additions for first round since there is no lag poly: gives a small speedup
            if self.current_round == 0 {
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

                let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                    LagrangePolynomial::new(&self.verifier_messages);
                for (_, _) in
                    Hypercube::<SignificantBitOrder, SignificantBitOrder>::new(self.current_round)
                {
                    let lag_poly = sequential_lag_poly.next().unwrap();
                    partial_sum_p_0 += self.stream_iterators[0].next().unwrap() * lag_poly;
                    partial_sum_q_0 += self.stream_iterators[1].next().unwrap() * lag_poly;
                }

                let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                    LagrangePolynomial::new(&self.verifier_messages);
                for (_, _) in
                    Hypercube::<SignificantBitOrder, SignificantBitOrder>::new(self.current_round)
                {
                    let lag_poly = sequential_lag_poly.next().unwrap();
                    partial_sum_p_1 += self.stream_iterators[0].next().unwrap() * lag_poly;
                    partial_sum_q_1 += self.stream_iterators[1].next().unwrap() * lag_poly;
                }

                sum_0 += partial_sum_p_0 * partial_sum_q_0;
                sum_1 += partial_sum_p_1 * partial_sum_q_1;
                sum_half +=
                    (partial_sum_p_0 + partial_sum_p_1) * (partial_sum_q_0 + partial_sum_q_1);
            }
        }
        sum_half *= self.inverse_four;
        (sum_0, sum_1, sum_half)
    }
}
