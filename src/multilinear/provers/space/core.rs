use ark_ff::Field;

use crate::{
    hypercube::Hypercube,
    interpolation::LagrangePolynomial,
    order_strategy::{GraycodeOrder, SignificantBitOrder},
    streams::Stream,
};

pub struct SpaceProver<F: Field, S: Stream<F>> {
    pub current_round: usize,
    pub evaluation_streams: Vec<S>,
    pub num_variables: usize,
    pub verifier_messages: Vec<F>,
    pub verifier_message_hats: Vec<F>,
}

impl<F: Field, S: Stream<F>> SpaceProver<F, S> {
    pub fn cty_evaluate(&self) -> (F, F) {
        // Initialize accumulators for sum_0 and sum_1
        let mut sum_0: F = F::ZERO;
        let mut sum_1: F = F::ZERO;

        // Create a bitmask for the number of free variables
        let bitmask: usize = 1 << (self.num_free_variables() - 1);

        // Iterate in two loops
        let num_vars_outer_loop = self.current_round;
        let num_vars_inner_loop = self.num_variables - num_vars_outer_loop;

        // Outer loop over a subset of variables
        for (index_outer, outer) in
            Hypercube::<GraycodeOrder, SignificantBitOrder>::new(num_vars_outer_loop)
        {
            // Calculate the weight using Lagrange polynomial
            let lag_poly: F = LagrangePolynomial::<F, GraycodeOrder>::lag_poly(
                self.verifier_messages.clone(),
                self.verifier_message_hats.clone(),
                outer,
            );

            if lag_poly == F::ZERO {
                // in this case the inner loop does nothing
                continue;
            }

            // Inner loop over all possible evaluations for the remaining variables
            for (index_inner, _inner) in
                Hypercube::<GraycodeOrder, SignificantBitOrder>::new(num_vars_inner_loop)
            {
                // Calculate the evaluation index
                let evaluation_index = index_outer << num_vars_inner_loop | index_inner;

                // Check if the bit at the position specified by the bitmask is set
                let is_set: bool = (evaluation_index & bitmask) != 0;

                for stream in &self.evaluation_streams {
                    // Use match to accumulate the appropriate value based on whether the bit is set or not
                    let inner_sum = stream.evaluation(evaluation_index) * lag_poly;
                    match is_set {
                        false => sum_0 += inner_sum,
                        true => sum_1 += inner_sum,
                    }
                }
            }
        }

        // Return the accumulated sums
        (sum_0, sum_1)
    }
    pub fn num_free_variables(&self) -> usize {
        self.num_variables - self.current_round
    }
    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
}
