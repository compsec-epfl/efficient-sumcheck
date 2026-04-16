use crate::{
    order_strategy::{AscendingOrder, MSBOrder, OrderStrategy},
    streams::Stream,
};
use ark_ff::Field;
use core::any::TypeId;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/*
 * It's totally reasonable to use this when the evaluations table
 * fits in memory (and yes, it's not so much a stream in this case)
 */

#[derive(Debug, Clone)]
pub struct MemoryStream<F: Field> {
    pub evaluations: Vec<F>,
}

/// Reorder `evaluations` according to the iteration order defined by `O`.
///
/// Fast paths for two well-known orders:
///   - [`MSBOrder`]: bit-reversal permutation, computed directly via
///     `usize::reverse_bits` and scattered in parallel with rayon. This is
///     the hot-path in recursive IOPs that pad + reorder at the entry of
///     each sumcheck call; at 2^24 it was measured at ~46% of total
///     sumcheck time in a prior profile.
///   - [`AscendingOrder`]: identity permutation — just returns `evaluations`
///     unchanged.
///
/// Arbitrary orders fall back to an iterator-based scatter (the original
/// generic path).
pub fn reorder_vec<F: Field, O: OrderStrategy + 'static>(evaluations: Vec<F>) -> Vec<F> {
    // abort if length not a power of two
    assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
    let num_vars = evaluations.len().trailing_zeros() as usize;

    // Fast path 1: MSB order is a bit-reversal permutation. Replace the
    // iterator-based scatter with hardware `reverse_bits` + parallel scatter.
    if TypeId::of::<O>() == TypeId::of::<MSBOrder>() {
        return bit_reverse_reorder(evaluations, num_vars);
    }

    // Fast path 2: AscendingOrder is the identity permutation. No reorder
    // needed — return the input unchanged.
    if TypeId::of::<O>() == TypeId::of::<AscendingOrder>() {
        return evaluations;
    }

    // Generic fallback: iterator-based scatter, one push per index.
    let mut order = O::new(num_vars);
    let mut evaluations_ordered = Vec::with_capacity(evaluations.len());
    for index in &mut order {
        evaluations_ordered.push(evaluations[index]);
    }
    evaluations_ordered
}

/// Below this input size, the bit-reverse scatter runs serially. Rayon's
/// fork/join overhead otherwise dominates the (very cheap) per-element
/// work at small n — measured at 3×+ slowdown vs serial for n = 2^16.
const BIT_REVERSE_PARALLEL_THRESHOLD: usize = 1 << 17;

/// Bit-reversal permutation: `out[i] = src[bit_reverse(i, num_vars)]`.
///
/// Uses `usize::reverse_bits` (hardware instruction on most targets) for the
/// index computation. Parallel-scatters via rayon above
/// `BIT_REVERSE_PARALLEL_THRESHOLD`; below that, runs serially to avoid
/// fork/join overhead.
#[inline]
fn bit_reverse_reorder<F: Field>(src: Vec<F>, num_vars: usize) -> Vec<F> {
    let n = src.len();
    if num_vars == 0 {
        // `reverse_bits() >> usize::BITS` is undefined behaviour; handle the
        // degenerate 1-element case (which is trivially identity) up front.
        return src;
    }
    let shift = usize::BITS - num_vars as u32;

    #[cfg(feature = "parallel")]
    {
        if n > BIT_REVERSE_PARALLEL_THRESHOLD {
            return (0..n)
                .into_par_iter()
                .map(|i| src[i.reverse_bits() >> shift])
                .collect();
        }
    }

    (0..n).map(|i| src[i.reverse_bits() >> shift]).collect()
}

impl<F: Field> MemoryStream<F> {
    pub fn new(evaluations: Vec<F>) -> Self {
        // abort if length not a power of two
        assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
        // return the MemoryStream instance
        Self { evaluations }
    }
    pub fn new_from_lex<O: OrderStrategy + 'static>(evaluations: Vec<F>) -> Self {
        // abort if length not a power of two
        assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
        Self::new(reorder_vec::<F, O>(evaluations))
    }
}

impl<F: Field> Stream<F> for MemoryStream<F> {
    fn evaluation(&self, point: usize) -> F {
        self.evaluations[point]
    }
    fn num_variables(&self) -> usize {
        self.evaluations.len().ilog2() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        order_strategy::{DescendingOrder, GraycodeOrder},
        tests::F64,
    };
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Iterator-based reference implementation — same shape as the original
    /// generic `reorder_vec` body before the TypeId fast paths were added.
    fn reorder_vec_iter_reference<F: Field, O: OrderStrategy>(evaluations: Vec<F>) -> Vec<F> {
        let num_vars = evaluations.len().trailing_zeros() as usize;
        let mut order = O::new(num_vars);
        let mut out = Vec::with_capacity(evaluations.len());
        for index in &mut order {
            out.push(evaluations[index]);
        }
        out
    }

    #[test]
    fn msb_fast_path_matches_iterator() {
        let mut rng = test_rng();
        for num_vars in [1usize, 2, 4, 8, 12] {
            let n = 1usize << num_vars;
            let input: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
            let expected = reorder_vec_iter_reference::<F64, MSBOrder>(input.clone());
            let got = reorder_vec::<F64, MSBOrder>(input);
            assert_eq!(got, expected, "mismatch at num_vars={}", num_vars);
        }
    }

    #[test]
    fn ascending_fast_path_is_identity() {
        let mut rng = test_rng();
        let n = 1usize << 8;
        let input: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let expected = input.clone();
        let got = reorder_vec::<F64, AscendingOrder>(input);
        assert_eq!(got, expected);
    }

    #[test]
    fn non_msb_fallback_still_works() {
        // Confirms the generic iterator path still runs correctly for
        // orders that don't have a fast path (Descending, Graycode).
        let mut rng = test_rng();
        let n = 1usize << 6;
        let input: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let expected_desc = reorder_vec_iter_reference::<F64, DescendingOrder>(input.clone());
        let got_desc = reorder_vec::<F64, DescendingOrder>(input.clone());
        assert_eq!(got_desc, expected_desc);

        let expected_gray = reorder_vec_iter_reference::<F64, GraycodeOrder>(input.clone());
        let got_gray = reorder_vec::<F64, GraycodeOrder>(input);
        assert_eq!(got_gray, expected_gray);
    }

    #[test]
    fn msb_num_vars_zero_edge_case() {
        // n = 1 (num_vars = 0) would trigger `x >> usize::BITS` UB if not
        // guarded. Confirm the short-circuit returns the input.
        let input = vec![F64::from(42u64)];
        let got = reorder_vec::<F64, MSBOrder>(input.clone());
        assert_eq!(got, input);
    }

    /// Ad-hoc timing comparison. Not a real benchmark — for a rough
    /// side-by-side of the new bit-reverse fast path vs the iterator
    /// reference. Run with:
    ///
    /// ```text
    /// cargo test --release --lib bench_reorder_msb -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn bench_reorder_msb() {
        use std::time::Instant;

        let mut rng = test_rng();
        for num_vars in [16usize, 18, 20, 22, 24] {
            let n = 1usize << num_vars;
            let input: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

            // Iterator reference (what was in the crate before this change).
            let clone = input.clone();
            let t0 = Instant::now();
            let _r1 = reorder_vec_iter_reference::<F64, MSBOrder>(clone);
            let t_iter = t0.elapsed();

            // Bit-reverse + parallel scatter fast path.
            let clone = input.clone();
            let t0 = Instant::now();
            let _r2 = reorder_vec::<F64, MSBOrder>(clone);
            let t_fast = t0.elapsed();

            let ratio = t_iter.as_secs_f64() / t_fast.as_secs_f64();
            println!(
                "num_vars={:>2}  n=2^{num_vars}  iter={:>10.3?}  fast={:>10.3?}  speedup={:.2}x",
                num_vars, t_iter, t_fast, ratio
            );
        }
    }
}
