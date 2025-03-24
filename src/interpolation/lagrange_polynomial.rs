use crate::{
    hypercube::{Hypercube, HypercubeMember},
    messages::VerifierMessages,
    order_strategy::{GraycodeOrder, OrderStrategy},
};
use ark_ff::Field;

#[derive(Debug)]
pub struct LagrangePolynomial<'a, F: Field, O: OrderStrategy> {
    order: O,
    last_position: usize,
    position: usize,
    value: F,
    verifier_messages: &'a VerifierMessages<F>,
    stop_position: usize,
}

impl<'a, F: Field, O: OrderStrategy> LagrangePolynomial<'a, F, O> {
    pub fn new(verifier_messages: &'a VerifierMessages<F>) -> Self {
        let num_vars = verifier_messages.messages.len();
        let order = O::new(num_vars);
        Self {
            order,
            last_position: 0,
            position: 0,
            value: verifier_messages.product_of_message_hats,
            verifier_messages,
            stop_position: Hypercube::<GraycodeOrder>::stop_value(num_vars),
        }
    }
    pub fn lag_poly(x: Vec<F>, x_hat: Vec<F>, b: HypercubeMember) -> F {
        // Iterate over the zipped triple x, x_hat, and boolean hypercube vectors
        x.iter().zip(x_hat.iter()).zip(b).fold(
            // Initial the accumulation to F::ONE
            F::ONE,
            // Closure for the folding operation, taking accumulator and ((x_i, x_hat_i), b_i)
            |acc, ((x_i, x_hat_i), b_i)| {
                // Multiply the accumulator by either x_i or x_hat_i based on the boolean value b_i
                acc * match b_i {
                    true => *x_i,
                    false => *x_hat_i,
                }
            },
        )
    }
    pub fn evaluate_from_three_points(verifier_message: F, prover_message: (F, F, F)) -> F {
        // Hardcoded x-values:
        let zero = F::zero();
        let one = F::one();
        let half = F::from(2_u32).inverse().unwrap();

        // Compute denominators for the Lagrange basis polynomials
        let inv_denom_0 = ((zero - one) * (zero - half)).inverse().unwrap();
        let inv_denom_1 = ((one - zero) * (one - half)).inverse().unwrap();
        let inv_denom_2 = ((half - zero) * (half - one)).inverse().unwrap();

        // Compute the Lagrange basis polynomials evaluated at x
        let basis_p_0 = (verifier_message - one) * (verifier_message - half) * inv_denom_0;
        let basis_p_1 = (verifier_message - zero) * (verifier_message - half) * inv_denom_1;
        let basis_p_2 = (verifier_message - zero) * (verifier_message - one) * inv_denom_2;

        // Return the evaluation of the unique quadratic polynomial
        prover_message.0 * basis_p_0 + prover_message.1 * basis_p_1 + prover_message.2 * basis_p_2
    }
}

impl<'a, F: Field, O: OrderStrategy> Iterator for LagrangePolynomial<'a, F, O> {
    type Item = F;
    fn next(&mut self) -> Option<Self::Item> {
        // Step 1: check if finished iterating
        if self.position >= self.stop_position {
            return None;
        }

        // Step 2: check if this iteration yields zero, in which case we skip processing
        let bit_agreement = !(self.verifier_messages.messages_zeros_and_ones_usize ^ self.position);
        if bit_agreement & self.verifier_messages.zero_ones_mask
            != self.verifier_messages.zero_ones_mask
        {
            // NOTICE! we do not update last_position in this case
            self.position = GraycodeOrder::next_gray_code(self.position);
            return Some(F::ZERO);
        }

        // Step 3: check if position is 0, which is a special case
        // Notice! step 2 could apply when position == 0
        if self.position == 0 {
            self.position = GraycodeOrder::next_gray_code(self.position);
            return Some(self.value);
        }

        // Step 4: update the value, skip if more than one bit difference
        let bit_diff = self.last_position ^ self.position;
        if bit_diff.count_ones() == 1 {
            let index_of_flipped_bit = bit_diff.trailing_zeros() as usize;
            let is_flipped_to_true = self.position & bit_diff != 0;
            let len = self.verifier_messages.messages.len();
            self.value = self.value
                * match is_flipped_to_true {
                    true => {
                        self.verifier_messages.message_and_message_hat_inverses
                            [len - index_of_flipped_bit - 1]
                    }
                    false => {
                        self.verifier_messages.message_hat_and_message_inverses
                            [len - index_of_flipped_bit - 1]
                    }
                };
        }

        // Step 5: increment positions
        self.last_position = self.position;
        self.position = GraycodeOrder::next_gray_code(self.position);

        // Step 6: return
        Some(self.value)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        hypercube::HypercubeMember, interpolation::LagrangePolynomial, messages::VerifierMessages,
        order_strategy::GraycodeOrder, tests::F19,
    };

    #[test]
    fn next() {
        // remember this is gray code ordering!
        let messages: Vec<F19> = vec![F19::from(13), F19::from(0), F19::from(7)];
        let message_hats: Vec<F19> = messages
            .clone()
            .iter()
            .map(|message| F19::from(1) - message)
            .collect();
        let vm = VerifierMessages::new(&vec![F19::from(13), F19::from(0), F19::from(7)]);
        let mut lag_poly: LagrangePolynomial<F19, GraycodeOrder> = LagrangePolynomial::new(&vm);
        for gray_code_index in [0, 1, 3, 2, 6, 7, 5, 4] {
            let exp = LagrangePolynomial::<F19, GraycodeOrder>::lag_poly(
                messages.clone(),
                message_hats.clone(),
                HypercubeMember::new(3, gray_code_index),
            );
            assert_eq!(lag_poly.next().unwrap(), exp);
        }
        assert_eq!(lag_poly.next(), None);
    }
    #[test]
    fn boolean_next() {
        // remember this is gray code ordering!
        let messages: Vec<F19> = vec![F19::from(0), F19::from(1), F19::from(1)];
        let message_hats: Vec<F19> = messages
            .clone()
            .iter()
            .map(|message| F19::from(1) - message)
            .collect();
        let vm = VerifierMessages::new(&vec![F19::from(0), F19::from(1), F19::from(1)]);
        let mut lag_poly: LagrangePolynomial<F19, GraycodeOrder> = LagrangePolynomial::new(&vm);
        for gray_code_index in [0, 1, 3, 2, 6, 7, 5, 4] {
            let exp = LagrangePolynomial::<F19, GraycodeOrder>::lag_poly(
                messages.clone(),
                message_hats.clone(),
                HypercubeMember::new(3, gray_code_index),
            );
            assert_eq!(lag_poly.next().unwrap(), exp);
        }
        assert_eq!(lag_poly.next(), None);
    }
}
