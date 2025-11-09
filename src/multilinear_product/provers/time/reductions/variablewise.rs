use ark_ff::Field;
use ark_std::cfg_into_iter;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::streams::Stream;

pub fn variablewise_product_evaluate<F: Field>(src: &[Vec<F>], inverse_four: F) -> (F, F, F) {
    let len = src[0].len();
    let second_half_bit: usize = len / 2;

    let p_evals = &src[0];
    let q_evals = &src[1];

    let acc00: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p0 = p_evals[i];
            let q0 = q_evals[i];
            p0 * q0
        })
        .sum();

    let acc11: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p1 = p_evals[i | second_half_bit];
            let q1 = q_evals[i | second_half_bit];
            p1 * q1
        })
        .sum();

    let acc01: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p0 = p_evals[i];
            let q1 = q_evals[i | second_half_bit];
            p0 * q1
        })
        .sum();

    let acc10: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p1 = p_evals[i | second_half_bit];
            let q0 = q_evals[i];
            p1 * q0
        })
        .sum();

    let sum_0 = acc00;
    let sum_1 = acc11;
    let mut sum_half = acc00 + acc11 + acc01 + acc10;
    sum_half *= inverse_four;

    (sum_0, sum_1, sum_half)
}

pub fn variablewise_product_evaluate_from_stream<F: Field, S: Stream<F>>(
    src: &[S],
    inverse_four: F,
) -> (F, F, F) {
    let len = 1usize << src[0].num_variables();
    let second_half_bit: usize = len / 2;

    let p_evals = &src[0];
    let q_evals = &src[1];

    let acc00: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p0 = p_evals.evaluation(i);
            let q0 = q_evals.evaluation(i);
            p0 * q0
        })
        .sum();

    let acc11: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p1 = p_evals.evaluation(i | second_half_bit);
            let q1 = q_evals.evaluation(i | second_half_bit);
            p1 * q1
        })
        .sum();

    let acc01: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p0 = p_evals.evaluation(i);
            let q1 = q_evals.evaluation(i | second_half_bit);
            p0 * q1
        })
        .sum();

    let acc10: F = cfg_into_iter!(0..second_half_bit)
        .map(|i| {
            let p1 = p_evals.evaluation(i | second_half_bit);
            let q0 = q_evals.evaluation(i);
            p1 * q0
        })
        .sum();

    let sum_0 = acc00;
    let sum_1 = acc11;
    let mut sum_half = acc00 + acc11 + acc01 + acc10;
    sum_half *= inverse_four;

    (sum_0, sum_1, sum_half)
}
