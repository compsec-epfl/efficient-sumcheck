use ark_ff::Field;
use ark_std::cfg_into_iter;

use crate::streams::Stream;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

pub fn variablewise_product_evaluate<F: Field>(src: &[Vec<F>], inverse_four: F) -> (F, F, F) {
    let len = src[0].len();
    let second_half_bit: usize = len / 2;
    // Initialize accumulators
    let mut sum_half = F::ZERO;
    let mut j_prime_table: ((F, F), (F, F)) = ((F::ZERO, F::ZERO), (F::ZERO, F::ZERO));
    let p_evals = &src[0];
    let q_evals = &src[1];

    let (acc00, acc01, acc10, acc11): (F, F, F, F) = cfg_into_iter!(0..second_half_bit)
        .into_par_iter()
        .fold(
            || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
            |(mut j00, mut j01, mut j10, mut j11), i| {
                let p0 = p_evals[i];
                let p1 = p_evals[i | second_half_bit];
                let q0 = q_evals[i];
                let q1 = q_evals[i | second_half_bit];

                j00 += p0 * q0; // (0,0)
                j11 += p1 * q1; // (1,1)
                j01 += p0 * q1; // (0,1)
                j10 += p1 * q0; // (1,0)

                (j00, j01, j10, j11)
            },
        )
        .reduce(
            || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
            |(a00, a01, a10, a11), (b00, b01, b10, b11)| {
                (a00 + b00, a01 + b01, a10 + b10, a11 + b11)
            },
        );

    j_prime_table.0 .0 += acc00;
    j_prime_table.0 .1 += acc01;
    j_prime_table.1 .0 += acc10;
    j_prime_table.1 .1 += acc11;

    // update
    let sum_0 = j_prime_table.0 .0;
    let sum_1 = j_prime_table.1 .1;
    sum_half += j_prime_table.0 .0 + j_prime_table.1 .1 + j_prime_table.0 .1 + j_prime_table.1 .0;
    sum_half *= inverse_four;

    (sum_0, sum_1, sum_half)
}

pub fn variablewise_product_evaluate_from_stream<F: Field, S: Stream<F>>(
    src: &[S],
    inverse_four: F,
) -> (F, F, F) {
    let len = 1usize << src[0].num_variables();
    let second_half_bit: usize = len / 2;
    // Initialize accumulators
    let mut sum_half = F::ZERO;
    let mut j_prime_table: ((F, F), (F, F)) = ((F::ZERO, F::ZERO), (F::ZERO, F::ZERO));
    let p_evals = &src[0];
    let q_evals = &src[1];

    let (acc00, acc01, acc10, acc11): (F, F, F, F) = cfg_into_iter!(0..second_half_bit)
        .into_par_iter()
        .fold(
            || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
            |(mut j00, mut j01, mut j10, mut j11), i| {
                let p0 = p_evals.evaluation(i);
                let p1 = p_evals.evaluation(i | second_half_bit);
                let q0 = q_evals.evaluation(i);
                let q1 = q_evals.evaluation(i | second_half_bit);
                j00 += p0 * q0; // (0,0)
                j11 += p1 * q1; // (1,1)
                j01 += p0 * q1; // (0,1)
                j10 += p1 * q0; // (1,0)

                (j00, j01, j10, j11)
            },
        )
        .reduce(
            || (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
            |(a00, a01, a10, a11), (b00, b01, b10, b11)| {
                (a00 + b00, a01 + b01, a10 + b10, a11 + b11)
            },
        );
    j_prime_table.0 .0 += acc00;
    j_prime_table.0 .1 += acc01;
    j_prime_table.1 .0 += acc10;
    j_prime_table.1 .1 += acc11;

    // update
    let sum_0 = j_prime_table.0 .0;
    let sum_1 = j_prime_table.1 .1;
    sum_half += j_prime_table.0 .0 + j_prime_table.1 .1 + j_prime_table.0 .1 + j_prime_table.1 .0;
    sum_half *= inverse_four;

    (sum_0, sum_1, sum_half)
}
