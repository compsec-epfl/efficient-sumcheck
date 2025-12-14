use ark_std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;

use ark_ff::Field;
use spongefish::{ProverState, codecs::arkworks_algebra::UnitToField};
use spongefish::codecs::arkworks_algebra::FieldToUnitSerialize;

use crate::{
    multilinear::{reductions::pairwise, ReduceMode},
    multilinear_product::{TimeProductProver, TimeProductProverConfig},
    prover::Prover,
    streams::MemoryStream,
    ProductSumcheck,
};

pub type FastMap<V> = HashMap<usize, V, BuildNoHashHasher<usize>>;

pub fn batched_constraint_poly<F: Field>(
    dense_polys: &Vec<Vec<F>>,
    sparse_polys: &FastMap<F>,
) -> Vec<F> {
    fn sum_columns<F: Field>(matrix: &Vec<Vec<F>>) -> Vec<F> {
        if matrix.is_empty() {
            return vec![];
        }
        let mut result = vec![F::ZERO; matrix[0].len()];
        for row in matrix {
            for (i, &val) in row.iter().enumerate() {
                result[i] += val;
            }
        }
        result
    }
    let mut res = sum_columns(dense_polys);
    for (k, v) in sparse_polys.iter() {
        res[*k] += v;
    }
    res
}

pub fn inner_product<F: Field>(
    f: &mut Vec<F>,
    g: &mut Vec<F>,
    prover_state: &mut ProverState,
) -> ProductSumcheck<F> {
    // checks
    assert_eq!(f.len(), g.len());
    assert!(f.len().count_ones() == 1);

    // initialize
    let num_rounds = f.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(F, F, F)> = vec![];
    let mut verifier_messages: Vec<F> = vec![];

    // all rounds
    for _ in 0..num_rounds {
        let mut prover = TimeProductProver::new(TimeProductProverConfig::new(
            f.len().trailing_zeros() as usize,
            vec![MemoryStream::new(f.to_vec()), MemoryStream::new(g.to_vec())],
            ReduceMode::Pairwise,
        ));

        // call the prover
        let msg = prover.next_message(None).unwrap();

        // write transcript
        prover_messages.push(msg);
        prover_state.add_scalars(&[msg.0, msg.1, msg.2]).unwrap();

        // read the transcript
        let [chg] = prover_state.challenge_scalars::<1>().unwrap();
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
