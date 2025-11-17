use ark_ff::Field;
use ark_std::vec::Vec;
use ark_std::{cfg_chunks, cfg_into_iter};
#[cfg(feature = "parallel")]
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    prelude::ParallelSlice,
};

use crate::streams::Stream;

pub fn evaluate<F: Field>(src: &[F]) -> (F, F) {
    let even_sum = cfg_into_iter!(0..src.len())
        .step_by(2)
        .map(|i| src[i])
        .sum();
    let odd_sum = cfg_into_iter!(1..src.len())
        .step_by(2)
        .map(|i| src[i])
        .sum();
    (even_sum, odd_sum)
}

pub fn evaluate_from_stream<F: Field, S: Stream<F>>(src: &S) -> (F, F) {
    let len = 1usize << src.num_variables();
    let even_sum = cfg_into_iter!(0..len)
        .step_by(2)
        .map(|i| src.evaluation(i))
        .sum();
    let odd_sum = cfg_into_iter!(1..len)
        .step_by(2)
        .map(|i| src.evaluation(i))
        .sum();
    (even_sum, odd_sum)
}

pub fn reduce_evaluations<F: Field>(src: &mut Vec<F>, verifier_message: F) {
    // compute from src
    let out: Vec<F> = cfg_chunks!(src, 2)
        .map(|chunk| chunk[0] + verifier_message * (chunk[1] - chunk[0]))
        .collect();
    // write back into src
    src[..out.len()].copy_from_slice(&out);
    src.truncate(out.len());
}

pub fn reduce_evaluations_from_stream<F: Field, S: Stream<F>>(
    src: &S,
    dst: &mut Vec<F>,
    verifier_message: F,
) {
    // compute from stream
    let len = 1usize << src.num_variables();
    let out: Vec<F> = cfg_into_iter!(0..len / 2)
        .map(|i| {
            let a = src.evaluation(2 * i);
            let b = src.evaluation((2 * i) + 1);
            a + verifier_message * (b - a)
        })
        .collect();
    *dst = out;
}
