use ark_ff::Field;
use ark_std::cfg_into_iter;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::streams::Stream;

/// Variablewise product evaluate returning coefficients `(a, b)`.
/// See [`pairwise_product_evaluate`](super::pairwise::pairwise_product_evaluate) for details.
pub fn variablewise_product_evaluate<F: Field>(src: &[Vec<F>]) -> (F, F) {
    let len = src[0].len();
    let second_half_bit: usize = len / 2;

    let p_evals = &src[0];
    let q_evals = &src[1];

    let a: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| p_evals[i] * q_evals[i])
        .sum();

    let b: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            p_evals[i] * q_evals[i | second_half_bit] + p_evals[i | second_half_bit] * q_evals[i]
        })
        .sum();

    (a, b)
}

/// Stream variant of [`variablewise_product_evaluate`].
pub fn variablewise_product_evaluate_from_stream<F: Field, S: Stream<F>>(src: &[S]) -> (F, F) {
    let len = 1usize << src[0].num_variables();
    let second_half_bit: usize = len / 2;

    let p_evals = &src[0];
    let q_evals = &src[1];

    let a: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| p_evals.evaluation(i) * q_evals.evaluation(i))
        .sum();

    let b: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            p_evals.evaluation(i) * q_evals.evaluation(i | second_half_bit)
                + p_evals.evaluation(i | second_half_bit) * q_evals.evaluation(i)
        })
        .sum();

    (a, b)
}
