use ark_ff::Field;
use ark_std::cfg_into_iter;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::streams::Stream;

pub fn pairwise_product_evaluate<F: Field>(src: &[Vec<F>]) -> (F, F, F) {
    let half_len = src[0].len() / 2;
    let sum00: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            let mut res = F::from(1);
            for p in src {
                res *= p[i];
            }
            res
        })
        .sum();

    let sum11: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            let mut res = F::from(1);
            for p in src {
                res *= p[i + 1];
            }
            res
        })
        .sum();

    let sum0110: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            let p0 = src[0][i];
            let p1 = src[0][i + 1];
            let q0 = src[1][i];
            let q1 = src[1][i + 1];
            p0 * q1 + p1 * q0
        })
        .sum();
    (sum00, sum11, sum0110)
}

pub fn pairwise_product_evaluate_from_stream<F: Field, S: Stream<F>>(src: &[S]) -> (F, F, F) {
    let len = 1usize << src[0].num_variables();
    let half_len = len / 2;
    let sum00: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            let p0 = src[0].evaluation(i);
            let q0 = src[1].evaluation(i);
            p0 * q0
        })
        .sum();

    let sum11: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            let p1 = src[0].evaluation(i + 1);
            let q1 = src[1].evaluation(i + 1);
            p1 * q1
        })
        .sum();

    let sum0110: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            let p0 = src[0].evaluation(i);
            let p1 = src[0].evaluation(i + 1);
            let q0 = src[1].evaluation(i);
            let q1 = src[1].evaluation(i + 1);
            p0 * q1 + p1 * q0
        })
        .sum();
    (sum00, sum11, sum0110)
}
