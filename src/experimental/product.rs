use ark_ff::Field;
use ark_std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;

use crate::{
    experimental::transcript::Transcript, multilinear::reductions::variablewise,
    multilinear_product::reductions::variablewise::variablewise_product_evaluate, ProductSumcheck,
};

pub type FastMap<V> = HashMap<usize, V, BuildNoHashHasher<usize>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MLEGroup<F: Field> {
    pub dense_polys: Vec<(Op, Vec<F>)>,
    pub sparse_polys: Vec<(Op, FastMap<F>)>,
}

pub fn reduce_sparse<F: Field>(src: &mut FastMap<F>, challenge: F) {
    let mut map = FastMap::default();
    for (&i, &eval) in src.iter() {
        *map.entry(i >> 1).or_insert(F::zero()) += eval
            * if i & 1 == 1 {
                challenge
            } else {
                F::one() - challenge
            };
    }
    *src = map;
}

fn collapse<F: Field>(group: &MLEGroup<F>) -> Vec<F> {
    let mut res = group.dense_polys.first().unwrap().1.clone();

    // handle all dense first?
    for i in 1..group.dense_polys.len() {
        let (op, v) = group.dense_polys.get(i).unwrap();
        for j in 0..res.len() {
            res[i] = match op {
                Op::Add => res[j] + v[j],
                Op::Sub => res[j] - v[j],
                Op::Mul => res[j] * v[j],
                Op::Div => res[j] / v[j],
            }
        }
    }

    // handle sparse
    for i in 0..group.sparse_polys.len() {
        let (op, v) = group.sparse_polys.get(i).unwrap();
        for (key, value) in v {
            res[*key] = match op {
                Op::Add => res[*key] + value,
                Op::Sub => res[*key] - value,
                Op::Mul => res[*key] * value,
                Op::Div => res[*key] / value,
            }
        }
    }

    res
}

fn reduce_group<F: Field>(group: &mut MLEGroup<F>, challenge: F) {
    for (_op, dense) in &mut group.dense_polys {
        variablewise::reduce_evaluations(dense, challenge, F::from(1) - challenge);
    }

    for (_op, sparse) in &mut group.sparse_polys {
        reduce_sparse(sparse, challenge);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckInput<F: Field> {
    pub mle_groups: Vec<MLEGroup<F>>,
    pub dense_mle_len: usize,
    pub num_vars: usize,
}

pub fn prove<F: Field>(
    input: &mut SumcheckInput<F>,
    prover_state: &mut impl Transcript<F>,
) -> ProductSumcheck<F> {
    let num_vars = input.num_vars;
    let mut prover_messages = vec![];
    let mut verifier_messages = vec![];
    // let mut new_evals = vec![];

    for i in 0..num_vars {
        // need to get each mle_group into one Vec<F>
        let mut eval_mles = Vec::with_capacity(input.mle_groups.len());
        for mle_group in &input.mle_groups {
            eval_mles.push(collapse(mle_group));
        }

        // evaluate
        let sums = variablewise_product_evaluate(&eval_mles, F::from(4).inverse().unwrap());
        prover_messages.push(sums);

        // absorb
        prover_state.write(sums.0);
        prover_state.write(sums.1);
        prover_state.write(sums.2);

        // reduce
        if i != input.num_vars - 1 {
            // squeeze
            let verifier_message = prover_state.read();
            verifier_messages.push(verifier_message);

            // for each of the polys in all of the poly groups reduce
            for mle_group in &mut input.mle_groups {
                reduce_group(mle_group, verifier_message);
            }
        }
    }

    ProductSumcheck::<F> {
        prover_messages,
        verifier_messages,
        is_accepted: true, // TODO: rm this
    }
}

#[cfg(test)]
mod tests {
    use crate::experimental::product::{prove, MLEGroup, Op, SumcheckInput};
    use crate::experimental::transcript::SanityTranscript;
    use crate::multilinear::ReduceMode;
    use crate::multilinear_product::{TimeProductProver, TimeProductProverConfig};
    use crate::prover::Prover;
    use crate::streams::Stream;
    use crate::tests::{BenchStream, M31};
    use crate::ProductSumcheck;

    #[test]
    fn sanity() {
        // check against this
        const NUM_VARIABLES: usize = 16;
        let evaluation_stream: BenchStream<M31> = BenchStream::new(NUM_VARIABLES);
        let time_prover_pairwise_transcript = ProductSumcheck::<M31>::prove::<
            BenchStream<M31>,
            TimeProductProver<M31, BenchStream<M31>>,
        >(
            &mut TimeProductProver::<M31, BenchStream<M31>>::new(TimeProductProverConfig::new(
                NUM_VARIABLES,
                vec![evaluation_stream.clone(), evaluation_stream.clone()],
                ReduceMode::Variablewise,
            )),
            &mut ark_std::test_rng(),
        );

        // sanity
        let s_evaluations: Vec<M31> = (0..1 << NUM_VARIABLES)
            .map(|i| evaluation_stream.evaluation(i))
            .collect();

        let f_mle_group = MLEGroup {
            dense_polys: vec![(Op::Mul, s_evaluations.clone())],
            sparse_polys: vec![],
        };

        let g_mle_group = MLEGroup {
            dense_polys: vec![(Op::Mul, s_evaluations.clone())],
            sparse_polys: vec![],
        };

        let mut sumcheck_input: SumcheckInput<M31> = SumcheckInput {
            mle_groups: vec![f_mle_group, g_mle_group],
            dense_mle_len: s_evaluations.len(),
            num_vars: s_evaluations.len().trailing_zeros() as usize,
        };

        let mut randomness = ark_std::test_rng();
        let mut transcript = SanityTranscript::new(&mut randomness);
        let transcript = prove(&mut sumcheck_input, &mut transcript);

        assert_eq!(time_prover_pairwise_transcript, transcript);
    }
}
