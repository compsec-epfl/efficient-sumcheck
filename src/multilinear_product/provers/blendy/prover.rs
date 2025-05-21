use ark_ff::Field;
use std::collections::BTreeSet;

use crate::{
    messages::VerifierMessages,
    multilinear_product::{BlendyProductProver, BlendyProductProverConfig, TimeProductProver},
    order_strategy::SignificantBitOrder,
    prover::Prover,
    streams::{Stream, StreamIterator},
};

impl<F: Field, S: Stream<F>> Prover<F> for BlendyProductProver<F, S> {
    type ProverConfig = BlendyProductProverConfig<F, S>;
    type ProverMessage = Option<(F, F, F)>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        let num_variables: usize = prover_config.num_variables;
        let num_stages: usize = prover_config.num_stages;
        let stage_size: usize = num_variables / num_stages;
        let max_rounds_phase2: usize = num_variables.div_ceil(2 * num_stages);

        let last_round_phase1: usize = 2;
        let last_round_phase3: usize = num_variables - num_variables.div_ceil(num_stages);

        let state_comp_set: BTreeSet<usize> = {
            let mut current_round: usize = last_round_phase1 + 1;
            let mut state_comp_set: BTreeSet<usize> = BTreeSet::new();
            while current_round <= last_round_phase3 {
                state_comp_set.insert(current_round);
                current_round =
                    std::cmp::min(current_round + max_rounds_phase2, current_round * 2 - 1); // the minus one is a time-efficiency optimization
                current_round = std::cmp::max(current_round, 2);
            }
            // println!("state_comp_set: {:?}", state_comp_set);
            state_comp_set
        };
        assert!(state_comp_set.len() > 0);

        let last_round: usize = *state_comp_set.iter().max().unwrap();
        let vsbw_prover = TimeProductProver::<F, S> {
            claim: prover_config.claim,
            current_round: 0,
            evaluations: vec![None; 2],
            streams: prover_config.streams.clone(),
            num_variables: num_variables - last_round + 1,
            inverse_four: F::from(4_u32).inverse().unwrap(),
        };

        let stream_iterators = prover_config
            .streams
            .iter()
            .cloned()
            .map(|s| StreamIterator::<F, S, SignificantBitOrder>::new(s))
            .collect();
        // return the BlendyProver instance
        Self {
            claim: prover_config.claim,
            current_round: 0,
            streams: prover_config.streams,
            stream_iterators,
            num_stages,
            num_variables,
            last_round_phase1,
            verifier_messages: VerifierMessages::new(&vec![]),
            verifier_messages_round_comp: VerifierMessages::new(&vec![]),
            x_table: vec![],
            y_table: vec![],
            j_prime_table: vec![],
            stage_size,
            inverse_four: F::from(4_u32).inverse().unwrap(),
            prev_table_round_num: 0,
            prev_table_size: 0,
            state_comp_set,
            switched_to_vsbw: false,
            vsbw_prover,
        }
    }

    fn next_message(&mut self, verifier_message: Self::VerifierMessage) -> Self::ProverMessage {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if !self.is_initial_round() {
            // this holds everything
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
            // this holds the randomness for between state computation r2
            self.verifier_messages_round_comp
                .receive_message(verifier_message.unwrap());
        }

        self.init_round_vars();

        self.compute_state();

        let sums: (F, F, F) = self.compute_round();

        // Increment the round counter
        self.current_round += 1;
        if self.switched_to_vsbw {
            self.vsbw_prover.current_round += 1;
        }
        // Return the computed polynomial sums
        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use ark_poly::multivariate::{SparsePolynomial, SparseTerm};

    use crate::{
        multilinear_product::{BlendyProductProver, BlendyProductProverConfig},
        order_strategy::SignificantBitOrder,
        prover::{ProductProverConfig, Prover},
        streams::{multivariate_product_claim, MemoryStream, Stream},
        tests::{
            multilinear_product::{consistency_test, BasicProductProver, BasicProductProverConfig},
            polynomials::Polynomial,
            BenchStream, F64,
        },
        ProductSumcheck,
    };

    // the stream has to be in SigBit order for this to work
    // #[test]
    // fn parity_with_basic_prover() {
    //     consistency_test::<F64, BenchStream<F64>, BlendyProductProver<F64, BenchStream<F64>>>();
    // }

    #[test]
    fn consistency_test_with_next_iterator() {
        // get evals in lexicographic order
        let num_variables = 8;
        let s_tmp: BenchStream<F64> = BenchStream::<F64>::new(num_variables).into();
        let mut evals: Vec<F64> = Vec::with_capacity(1 << num_variables);
        for i in 0..(1 << num_variables) {
            evals.push(s_tmp.evaluation(i));
        }

        // create the stream in SigBit order
        let s: MemoryStream<F64> =
            MemoryStream::new_from_lex::<SignificantBitOrder>(evals.clone()).into();
        let claim: F64 = multivariate_product_claim(vec![s.clone(), s.clone()]);

        // get transcript from Blendy prover
        let prover_transcript: ProductSumcheck<F64> = ProductSumcheck::<F64>::prove::<
            MemoryStream<F64>,
            BlendyProductProver<F64, MemoryStream<F64>>,
        >(
            &mut Prover::<F64>::new(BlendyProductProverConfig::default(
                claim,
                num_variables,
                vec![s.clone(), s],
            )),
            &mut ark_std::test_rng(),
        );

        // get transcript from SanityProver
        let p: SparsePolynomial<F64, SparseTerm> =
            <SparsePolynomial<F64, SparseTerm> as Polynomial<F64>>::from_hypercube_evaluations(
                evals,
            );
        let mut sanity_prover = BasicProductProver::<F64>::new(BasicProductProverConfig::new(
            claim.clone(),
            num_variables,
            p.clone(),
            p,
        ));
        let sanity_prover_transcript = ProductSumcheck::<F64>::prove::<
            MemoryStream<F64>,
            BasicProductProver<F64>,
        >(&mut sanity_prover, &mut ark_std::test_rng());

        // ensure the transcript is identical
        assert_eq!(prover_transcript.is_accepted, true);
        assert_eq!(prover_transcript, sanity_prover_transcript);
    }
}
