use ark_ff::Field;

#[derive(Clone, Debug)]
pub struct VerifierMessages<F: Field> {
    pub messages: Vec<F>,
    pub message_hats: Vec<F>,
    pub message_and_message_hat_inverses: Vec<F>,
    pub message_hat_and_message_inverses: Vec<F>,
    pub messages_zeros_and_ones_usize: usize,
    pub zero_ones_mask: usize,
    pub product_of_message_hats: F,
}

impl<F: Field> VerifierMessages<F> {
    pub fn new(messages: &Vec<F>) -> Self {
        let mut verifier_messages = Self {
            messages: vec![],
            message_hats: vec![],
            product_of_message_hats: F::ONE,
            message_and_message_hat_inverses: vec![],
            message_hat_and_message_inverses: vec![],
            messages_zeros_and_ones_usize: 0,
            zero_ones_mask: 0,
        };
        for message in messages {
            verifier_messages.receive_message(*message);
        }
        verifier_messages
    }
    pub fn new_from_self(vm: &Self, start: usize, end: usize) -> Self {
        // TODO (z-tech): this can be redone more efficiently
        Self::new(&vm.messages[start..end].to_vec())
    }
    pub fn receive_message(&mut self, message: F) {
        // Step 1: compute some things
        let message_hat = F::ONE - message;
        let message_inverse = match message.inverse() {
            Some(inverse) => inverse,
            None => F::ONE,
        };
        let message_hat_inverse = match message_hat.inverse() {
            Some(inverse) => inverse,
            None => F::ONE,
        };
        // Step 2: store some things
        self.messages.push(message);
        self.message_hats.push(message_hat);
        self.message_and_message_hat_inverses
            .push(message * message_hat_inverse);
        self.message_hat_and_message_inverses
            .push(message_hat * message_inverse);

        if message == F::ZERO || message_hat == F::ZERO {
            self.zero_ones_mask = (self.zero_ones_mask << 1) | 1;
            self.messages_zeros_and_ones_usize = if message == F::ONE {
                self.messages_zeros_and_ones_usize << 1 | 1
            } else {
                self.messages_zeros_and_ones_usize << 1
            };
        } else {
            self.zero_ones_mask = self.zero_ones_mask << 1;
            self.messages_zeros_and_ones_usize = self.messages_zeros_and_ones_usize << 1;
            self.product_of_message_hats = self.product_of_message_hats * message_hat;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{messages::VerifierMessages, tests::F19};
    use ark_ff::{One, Zero};

    #[test]
    fn receive_message() {
        let mut m0 = VerifierMessages::new(&vec![]);

        // ## receive 13
        m0.receive_message(F19::from(13));
        assert_eq!(m0.messages, vec![F19::from(13)]);
        assert_eq!(m0.message_hats, vec![F19::one() - F19::from(13)]);

        // ## receive 0
        m0.receive_message(F19::zero());
        assert_eq!(m0.messages, vec![F19::from(13), F19::zero()]);
        assert_eq!(
            m0.message_hats,
            vec![F19::one() - F19::from(13), F19::one()]
        );

        // ## receive 7
        m0.receive_message(F19::from(7));
        assert_eq!(m0.messages, vec![F19::from(13), F19::zero(), F19::from(7)]);
        assert_eq!(
            m0.message_hats,
            vec![
                F19::one() - F19::from(13),
                F19::one(),
                F19::one() - F19::from(7)
            ]
        );

        // ## receive 1
        m0.receive_message(F19::one());
        assert_eq!(
            m0.messages,
            vec![F19::from(13), F19::zero(), F19::from(7), F19::one()]
        );
        assert_eq!(
            m0.message_hats,
            vec![
                F19::one() - F19::from(13),
                F19::one(),
                F19::one() - F19::from(7),
                F19::zero()
            ]
        );

        let mut m1 = VerifierMessages::new(&vec![]);

        // ## receive zero
        m1.receive_message(F19::from(0));
        assert_eq!(m1.messages, vec![F19::from(0)]);
        assert_eq!(m1.message_hats, vec![F19::one()]);

        // receive 1
        m1.receive_message(F19::from(1));
        assert_eq!(m1.messages, vec![F19::from(0), F19::one()]);
        assert_eq!(m1.message_hats, vec![F19::one(), F19::zero()]);
    }
}
