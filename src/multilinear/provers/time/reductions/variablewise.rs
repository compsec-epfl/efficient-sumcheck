use ark_ff::Field;
use ark_std::vec::Vec;
use ark_std::{cfg_into_iter, cfg_iter_mut};
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::streams::Stream;

pub fn evaluate<F: Field>(src: &[F], num_free_variables: usize) -> (F, F) {
    let bitmask: usize = 1 << (num_free_variables - 1);
    let (sum_0, sum_1) = cfg_into_iter!(0..src.len())
        .map(|i| {
            let val = src[i];
            // Route value into the proper bucket
            if (i & bitmask) == 0 {
                (val, F::zero()) // contributes to sum_0
            } else {
                (F::zero(), val) // contributes to sum_1
            }
        })
        // Combine partial (sum0, sum1) pairs from each worker/thread.
        .reduce(
            || (F::zero(), F::zero()),
            |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
        );
    (sum_0, sum_1)
}

pub fn evaluate_from_stream<F: Field, S: Stream<F>>(src: &S, num_free_variables: usize) -> (F, F) {
    let len = 1usize << src.num_variables();
    let bitmask: usize = 1 << (num_free_variables - 1);
    let (sum_0, sum_1) = cfg_into_iter!(0..len)
        .map(|i| {
            let val = src.evaluation(i);
            // Route value into the proper bucket
            if (i & bitmask) == 0 {
                (val, F::zero()) // contributes to sum_0
            } else {
                (F::zero(), val) // contributes to sum_1
            }
        })
        // Combine partial (sum0, sum1) pairs from each worker/thread.
        .reduce(
            || (F::zero(), F::zero()),
            |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
        );
    (sum_0, sum_1)
}

pub fn reduce_evaluations<F: Field>(
    src: &mut Vec<F>,
    num_free_variables: usize,
    verifier_message: F,
    verifier_message_hat: F,
) {
    let setbit: usize = 1 << num_free_variables;
    let mut dest = vec![F::zero(); src.len() / 2];
    cfg_iter_mut!(dest).enumerate().for_each(|(i0, slot)| {
        let i1 = i0 | setbit;
        let v0 = src[i0];
        let v1 = src[i1];
        *slot = v0 * verifier_message_hat + v1 * verifier_message;
    });
    *src = dest;
}

pub fn reduce_evaluations_from_stream<F: Field, S: Stream<F>>(
    src: &S,
    dst: &mut Vec<F>,
    num_free_variables: usize,
    verifier_message: F,
    verifier_message_hat: F,
) {
    // compute from stream
    let len = 2_i32.pow(src.num_variables() as u32) as usize;
    let setbit: usize = 1 << num_free_variables;
    let mut out = vec![F::zero(); len / 2];
    cfg_iter_mut!(out).enumerate().for_each(|(i0, slot)| {
        let i1 = i0 | setbit;
        let v0 = src.evaluation(i0);
        let v1 = src.evaluation(i1);
        *slot = v0 * verifier_message_hat + v1 * verifier_message;
    });
    // write back into dst
    if dst.len() < out.len() {
        dst.resize(out.len(), F::zero());
    }
    dst[..out.len()].copy_from_slice(&out);
}
