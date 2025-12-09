use ark_ff::Field;
use ark_std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;

use crate::{ProductSumcheck, multilinear_product::reductions::pairwise::pairwise_product_evaluate};

pub type FastMap<V> = HashMap<usize, V, BuildNoHashHasher<usize>>;

pub trait FiatShamir<T> {
    fn absorb(&mut self, value: T);
    fn squeeze(&mut self) -> T;
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckInput<F: Field> {
    pub mle_groups: Vec<MLEGroup<F>>,
    pub dense_mle_len: usize,
    pub num_vars: usize,
}

pub fn prove<F: Field>(input: SumcheckInput<F>, prover_state: &mut impl FiatShamir<F>) -> ProductSumcheck<F> {
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
        let sums = pairwise_product_evaluate(&eval_mles);
        prover_messages.push(sums);

        // absorb
        prover_state.absorb(sums.0);
        prover_state.absorb(sums.1);
        prover_state.absorb(sums.2);

        // // reduce
        // if i != input.num_vars - 1 {
        //     // squeeze
        //     let verifier_message = prover_state.squeeze();
        //     verifier_messages.push(verifier_message);
        //     // reduce
        //     reduce_ef(&mut new_evals, verifier_message);
        // }
    }

    ProductSumcheck::<F> {
        prover_messages,
        verifier_messages,
        is_accepted: true, // TODO: rm this
    }
    
}