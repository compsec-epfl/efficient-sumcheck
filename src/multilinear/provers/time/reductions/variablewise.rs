use ark_ff::Field;
#[cfg(feature = "parallel")]
use ark_std::cfg_into_iter;
use ark_std::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::streams::Stream;

pub fn evaluate<F: Field>(src: &[F]) -> (F, F) {
    let second_half_bit: usize = src.len() / 2;

    #[cfg(feature = "parallel")]
    let (sum_0, sum_1) = cfg_into_iter!(0..src.len())
        .map(|i| {
            let v = src[i];
            if (i & second_half_bit) == 0 {
                (v, F::zero())
            } else {
                (F::zero(), v)
            }
        })
        .reduce(
            || (F::zero(), F::zero()),
            |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
        );

    #[cfg(not(feature = "parallel"))]
    let (sum_0, sum_1) = {
        let mut sum_0 = F::ZERO;
        let mut sum_1 = F::ZERO;
        for i in 0..src.len() {
            let v = src[i];
            match i & second_half_bit != 0 {
                false => sum_0 += v,
                true => sum_1 += v,
            }
        }
        (sum_0, sum_1)
    };

    (sum_0, sum_1)
}

pub fn evaluate_from_stream<F: Field, S: Stream<F>>(src: &S) -> (F, F) {
    let len = 1usize << src.num_variables();
    let second_half_bit: usize = len / 2;

    #[cfg(feature = "parallel")]
    let (sum_0, sum_1) = cfg_into_iter!(0..len)
        .map(|i| {
            let v = src.evaluation(i);
            if (i & second_half_bit) == 0 {
                (v, F::zero())
            } else {
                (F::zero(), v)
            }
        })
        .reduce(
            || (F::zero(), F::zero()),
            |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
        );

    #[cfg(not(feature = "parallel"))]
    let (sum_0, sum_1) = {
        let mut sum_0 = F::ZERO;
        let mut sum_1 = F::ZERO;
        for i in 0..len {
            let v = src.evaluation(i);
            match i & second_half_bit != 0 {
                false => sum_0 += v,
                true => sum_1 += v,
            }
        }
        (sum_0, sum_1)
    };

    (sum_0, sum_1)
}

pub fn reduce_evaluations<F: Field>(
    src: &mut Vec<F>,
    verifier_message: F,
    verifier_message_hat: F,
) {
    let second_half_bit: usize = src.len() / 2;
    let mut out = vec![F::ZERO; src.len() / 2];

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut()
            .enumerate()
            .for_each(|(first_half_index, slot): (usize, &mut F)| {
                let second_half_index = first_half_index | second_half_bit;
                let v0 = src[first_half_index];
                let v1 = src[second_half_index];
                *slot = v0 * verifier_message_hat + v1 * verifier_message;
            });
    }

    #[cfg(not(feature = "parallel"))]
    for first_half_index in 0..src.len() / 2 {
        let second_half_index = first_half_index | second_half_bit;
        let v0 = src[first_half_index];
        let v1 = src[second_half_index];
        out[first_half_index] = v0 * verifier_message_hat + v1 * verifier_message;
    }

    *src = out;
}

pub fn reduce_evaluations_from_stream<F: Field, S: Stream<F>>(
    src: &S,
    dst: &mut Vec<F>,
    verifier_message: F,
    verifier_message_hat: F,
) {
    let len = 1usize << src.num_variables();
    let second_half_bit: usize = len / 2;
    let mut out = vec![F::ZERO; len / 2];

    #[cfg(feature = "parallel")]
    out.par_iter_mut()
        .enumerate()
        .for_each(|(first_half_index, slot): (usize, &mut F)| {
            let second_half_index = first_half_index | second_half_bit;
            let v0 = src.evaluation(first_half_index);
            let v1 = src.evaluation(second_half_index);
            *slot = v0 * verifier_message_hat + v1 * verifier_message;
        });

    #[cfg(not(feature = "parallel"))]
    for first_half_index in 0..len / 2 {
        let second_half_index = first_half_index | second_half_bit;
        let v0 = src.evaluation(first_half_index);
        let v1 = src.evaluation(second_half_index);
        out[first_half_index] = v0 * verifier_message_hat + v1 * verifier_message;
    }

    *dst = out;
}
