use ark_ff::Field;
use ark_std::cfg_into_iter;
use ark_std::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::streams::Stream;

pub fn evaluate<F: Field>(src: &[F]) -> (F, F) {
    let sum_0 = cfg_into_iter!(0..src.len() / 2).map(|i| src[i]).sum();
    let sum_1 = cfg_into_iter!(src.len() / 2..src.len())
        .map(|i| src[i])
        .sum();
    (sum_0, sum_1)
}

pub fn evaluate_from_stream<F: Field, S: Stream<F>>(src: &S) -> (F, F) {
    let len = 1usize << src.num_variables();
    let sum_0 = cfg_into_iter!(0..len / 2).map(|i| src.evaluation(i)).sum();
    let sum_1 = cfg_into_iter!(len / 2..len)
        .map(|i| src.evaluation(i))
        .sum();
    (sum_0, sum_1)
}

pub fn reduce_evaluations<F: Field>(src: &mut Vec<F>, verifier_message: F) {
    let second_half_bit: usize = src.len() / 2;
    let out: Vec<F> = cfg_into_iter!(0..src.len() / 2)
        .map(|i0| {
            let v0 = src[i0];
            let i1 = i0 | second_half_bit;
            let v1 = src[i1];
            v0 + verifier_message * (v1 - v0)
        })
        .collect();
    *src = out;
}

pub fn reduce_evaluations_from_stream<F: Field, S: Stream<F>>(
    src: &S,
    dst: &mut Vec<F>,
    verifier_message: F,
) {
    let len = 1usize << src.num_variables();
    let second_half_bit: usize = len / 2;
    let out: Vec<F> = cfg_into_iter!(0..len / 2)
        .map(|i0| {
            let v0 = src.evaluation(i0);
            let i1 = i0 | second_half_bit;
            let v1 = src.evaluation(i1);
            v0 + verifier_message * (v1 - v0)
        })
        .collect();
    *dst = out;
}
