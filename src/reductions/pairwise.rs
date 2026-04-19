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
    /// Below this input size, the serial in-place path wins: rayon's
    /// fork/join overhead exceeds the actual compute, and we avoid the
    /// `.collect()` allocation entirely. Above it, parallelism outpaces
    /// serial even with the allocation cost. Chosen to match typical L1
    /// cache-blocking on modern SIMD hosts (~4K field elements).
    const SERIAL_THRESHOLD: usize = 1 << 12;

    #[cfg(feature = "parallel")]
    {
        if src.len() > SERIAL_THRESHOLD {
            // Parallel path: MSB pairing makes true in-place parallel
            // impossible without unsafe (writer i writes src[i] while writer
            // j reads src[2j] which may alias src[i]). Allocate a fresh Vec
            // via rayon's parallel collect; `*src = out` swaps buffers
            // without a copy.
            let out: Vec<F> = cfg_chunks!(src, 2)
                .map(|chunk| chunk[0] + verifier_message * (chunk[1] - chunk[0]))
                .collect();
            *src = out;
            return;
        }
    }

    // Serial path: truly in-place. Writing src[i] while reading src[2i] and
    // src[2i+1] is safe sequentially because 2i ≥ i always, so we never
    // clobber a read we still need. Used for non-parallel builds and for
    // small inputs where rayon overhead would dominate.
    let new_len = src.len() / 2;
    for i in 0..new_len {
        let a = src[2 * i];
        let b = src[2 * i + 1];
        src[i] = a + verifier_message * (b - a);
    }
    src.truncate(new_len);
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

/// Pairwise product evaluate returning coefficients `(a, b)` of the degree-2
/// round polynomial `q(x) = a + bx + cx²`:
///   - `a = Σ f_even · g_even`
///   - `b = Σ (f_even · g_odd + f_odd · g_even)`
pub fn pairwise_product_evaluate<F: Field>(src: &[Vec<F>]) -> (F, F) {
    let half_len = src[0].len() / 2;
    let a: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            src[0][i] * src[1][i]
        })
        .sum();
    let b: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            src[0][i] * src[1][i + 1] + src[0][i + 1] * src[1][i]
        })
        .sum();
    (a, b)
}

/// Cross-field reduce: fold `BF` evaluations with an `EF` challenge, producing `Vec<EF>`.
///
/// For each adjacent pair `(a, b)` in `src`: `EF::from(a) + challenge * (EF::from(b) - EF::from(a))`.
pub fn cross_field_reduce<BF: Field, EF: Field + From<BF>>(src: &[BF], challenge: EF) -> Vec<EF> {
    cfg_chunks!(src, 2)
        .map(|chunk| {
            let a = EF::from(chunk[0]);
            let b = EF::from(chunk[1]);
            a + challenge * (b - a)
        })
        .collect()
}
