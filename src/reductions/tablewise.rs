use crate::field::SumcheckRing;
use ark_std::{cfg_chunks, vec::Vec};
#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, prelude::ParallelSlice};

pub fn reduce_evaluations<F: SumcheckRing>(src: &mut Vec<Vec<F>>, verifier_message: F) {
    let out: Vec<Vec<F>> = cfg_chunks!(src, 2)
        .map(|chunk| {
            chunk[0]
                .iter()
                .zip(&chunk[1])
                .map(|(&a, &b)| a + verifier_message * (b - a))
                .collect::<Vec<F>>()
        })
        .collect();
    *src = out;
}
