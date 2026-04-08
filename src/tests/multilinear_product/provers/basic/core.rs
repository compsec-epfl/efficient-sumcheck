use ark_ff::Field;
use ark_poly::multivariate::{SparsePolynomial, SparseTerm};

use crate::{
    hypercube::Hypercube, messages::VerifierMessages, order_strategy::GraycodeOrder,
    tests::polynomials::Polynomial,
};
pub struct BasicProductProver<F: Field> {
    pub current_round: usize,
    pub inverse_four: F,
    pub num_variables: usize,
    pub p: SparsePolynomial<F, SparseTerm>,
    pub q: SparsePolynomial<F, SparseTerm>,
    pub verifier_messages: VerifierMessages<F>,
}

impl<F: Field> BasicProductProver<F> {
    /// Returns `(a, b)` — the constant and linear coefficients of the degree-2
    /// round polynomial `q(x) = a + bx + cx²`.
    pub fn compute_round(&self) -> (F, F) {
        let mut a = F::ZERO; // sum of p0*q0 (even-even)
        let mut b = F::ZERO; // sum of p0*q1 + p1*q0 (cross-term)
        for (_, hypercube_member) in
            Hypercube::<GraycodeOrder>::new(self.num_variables - self.current_round - 1)
        {
            let partial_point: Vec<F> = hypercube_member
                .to_vec_bool()
                .into_iter()
                .map(|bit: bool| -> F {
                    if bit {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                })
                .collect();
            let partial_point_zero: Vec<F> = std::iter::once(F::ZERO)
                .chain(partial_point.iter().cloned())
                .collect();
            let partial_point_one: Vec<F> = std::iter::once(F::ONE)
                .chain(partial_point.iter().cloned())
                .collect();
            let point_zero: Vec<F> = self
                .verifier_messages
                .messages
                .iter()
                .cloned()
                .chain(partial_point_zero.iter().cloned())
                .collect();
            let point_one: Vec<F> = self
                .verifier_messages
                .messages
                .iter()
                .cloned()
                .chain(partial_point_one.iter().cloned())
                .collect();
            let p_zero = self.p.evaluate(point_zero.clone()).unwrap();
            let p_one = self.p.evaluate(point_one.clone()).unwrap();
            let q_zero = self.q.evaluate(point_zero.clone()).unwrap();
            let q_one = self.q.evaluate(point_one.clone()).unwrap();
            a += p_zero * q_zero;
            b += p_zero * q_one + p_one * q_zero;
        }
        (a, b)
    }
    pub fn is_initial_round(&self) -> bool {
        self.current_round == 0
    }
    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
}
