use ark_ff::Field;
use ark_std::cfg_into_iter;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::streams::Stream;

/// Pairwise product evaluate returning coefficients `(a, b)` of the degree-2
/// round polynomial `q(x) = a + bx + cx²`:
///   - `a = Σ f_even · g_even`  (constant coefficient, = q(0))
///   - `b = Σ (f_even · g_odd + f_odd · g_even)`  (linear coefficient)
///
/// The quadratic coefficient `c = Σ f_odd · g_odd` is NOT returned; the
/// verifier derives it as `c = claim - 2a - b`.
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

/// Stream variant of [`pairwise_product_evaluate`].
pub fn pairwise_product_evaluate_from_stream<F: Field, S: Stream<F>>(src: &[S]) -> (F, F) {
    let len = 1usize << src[0].num_variables();
    let half_len = len / 2;
    let a: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            src[0].evaluation(i) * src[1].evaluation(i)
        })
        .sum();

    let b: F = cfg_into_iter!(0..half_len)
        .map(|k| {
            let i = 2 * k;
            src[0].evaluation(i) * src[1].evaluation(i + 1)
                + src[0].evaluation(i + 1) * src[1].evaluation(i)
        })
        .sum();
    (a, b)
}
