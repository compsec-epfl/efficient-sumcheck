use ark_ff::Field;
#[cfg(feature = "parallel")]
use ark_std::{cfg_chunks, vec::Vec};
#[cfg(feature = "parallel")]
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    prelude::ParallelSlice,
};

pub fn reduce_evaluations<F: Field>(src: &mut Vec<Vec<F>>, verifier_message: F) {
    let out: Vec<Vec<F>> = cfg_chunks!(src, 2)
        .into_par_iter()
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
