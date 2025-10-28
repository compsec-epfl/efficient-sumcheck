use ark_ff::Field;
use ark_std::vec::Vec;
use ark_std::{cfg_chunks, cfg_into_iter, cfg_iter};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    prelude::ParallelSlice,
};

use crate::streams::Stream;

pub fn evaluate<F: Field>(src: &Vec<F>) -> (F, F) {
    // Sum even indices
    let even_sum = cfg_iter!(src)
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, val)| *val)
        .sum();

    // Sum odd indices
    let odd_sum = cfg_iter!(src)
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, val)| *val)
        .sum();

    (even_sum, odd_sum)
}

pub fn evaluate_from_stream<F: Field, S: Stream<F>>(src: &S) -> (F, F) {
    let len = 1usize << src.num_variables();

    cfg_into_iter!(0..len)
        .map(|i| (i, src.evaluation(i)))
        .fold(
            || (F::zero(), F::zero()),
            |(mut even, mut odd), (i, val)| {
                if i % 2 == 0 {
                    even += val;
                } else {
                    odd += val;
                }
                (even, odd)
            },
        )
        .reduce(
            || (F::zero(), F::zero()),
            |(e1, o1), (e2, o2)| (e1 + e2, o1 + o2),
        )
}

pub fn reduce_evaluations<F: Field>(
    src: &mut Vec<F>,
    verifier_message: F,
    verifier_message_hat: F,
) {
    // compute from src
    let out: Vec<F> = cfg_chunks!(src, 2)
        .map(|chunk| chunk[0] * verifier_message_hat + chunk[1] * verifier_message)
        .collect();
    // write back into src
    src[..out.len()].copy_from_slice(&out);
    src.truncate(out.len());
}

pub fn reduce_evaluations_from_stream<F: Field, S: Stream<F>>(
    stream: &S,
    dst: &mut Vec<F>,
    verifier_message: F,
    verifier_message_hat: F,
) {
    // compute from stream
    let len = 2_i32.pow(stream.num_variables() as u32) as usize;
    let out: Vec<F> = cfg_into_iter!(0..len / 2)
        .map(|i| {
            let a = stream.evaluation(2 * i);
            let b = stream.evaluation((2 * i) + 1);
            a * verifier_message_hat + b * verifier_message
        })
        .collect();
    // write back into dst
    if dst.len() < out.len() {
        dst.resize(out.len(), F::zero());
    }
    dst[..out.len()].copy_from_slice(&out);
}
