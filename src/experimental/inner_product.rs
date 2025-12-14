use std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;

use ark_ff::Field;

use crate::{
    ProductSumcheck, experimental::transcript::Transcript, multilinear::{ReduceMode, reductions::pairwise}, multilinear_product::{TimeProductProver, TimeProductProverConfig}, prover::Prover, streams::MemoryStream
};

pub type FastMap<V> = HashMap<usize, V, BuildNoHashHasher<usize>>;

pub fn inner_product<F: Field>(
    f: &mut Vec<F>,
    g: &mut Vec<F>,
    prover_state: &mut impl Transcript<F>,
) -> ProductSumcheck<F> {
    // checks
    assert_eq!(f.len(), g.len());
    assert!(f.len().count_ones() == 1);

    // initialize
    let num_vars = f.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(F, F, F)> = vec![];
    let mut verifier_messages: Vec<F> = vec![];

    // all other rounds
    for round_num in (0..num_vars).rev() {
        let mut prover = TimeProductProver::new(TimeProductProverConfig::new(
            round_num,
            vec![MemoryStream::new(f.to_vec()), MemoryStream::new(g.to_vec())],
            ReduceMode::Pairwise,
        ));

        // call the prover
        let msg = prover.next_message(None).unwrap();

        // write transcript
        prover_messages.push(msg);
        prover_state.write(msg.0);
        prover_state.write(msg.1);
        prover_state.write(msg.2);

        // read the transcript
        let chg = prover_state.read();
        verifier_messages.push(chg);

        // reduce
        pairwise::reduce_evaluations(f, chg);
        pairwise::reduce_evaluations(g, chg);
    }

    ProductSumcheck {
        verifier_messages,
        prover_messages,
    }
}
