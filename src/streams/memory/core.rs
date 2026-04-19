use crate::streams::Stream;
use ark_ff::Field;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// In-memory evaluation stream.
#[derive(Debug, Clone)]
pub struct MemoryStream<F: Field> {
    pub evaluations: Vec<F>,
}

/// Bit-reversal permutation: reorders evaluations from ascending (lexicographic)
/// to MSB (half-split) layout.
///
/// `out[i] = src[bit_reverse(i, num_vars)]`.
///
/// Uses `usize::reverse_bits` (hardware instruction on most targets).
/// Parallel via rayon above the threshold.
pub fn reorder_vec_msb<F: Field>(evaluations: Vec<F>) -> Vec<F> {
    assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
    let num_vars = evaluations.len().trailing_zeros() as usize;
    bit_reverse_reorder(evaluations, num_vars)
}

const BIT_REVERSE_PARALLEL_THRESHOLD: usize = 1 << 17;

#[inline]
fn bit_reverse_reorder<F: Field>(src: Vec<F>, num_vars: usize) -> Vec<F> {
    let n = src.len();
    if num_vars == 0 {
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
        assert!(!evaluations.is_empty() && evaluations.len().count_ones() == 1);
        Self { evaluations }
    }

    /// Construct from ascending (lex) order evaluations, reordering to MSB.
    pub fn new_from_lex_msb(evaluations: Vec<F>) -> Self {
        Self::new(reorder_vec_msb(evaluations))
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
    use crate::tests::F64;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn msb_reorder_roundtrip() {
        let mut rng = test_rng();
        for num_vars in [1usize, 2, 4, 8, 12] {
            let n = 1usize << num_vars;
            let input: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
            // Double bit-reverse should be identity.
            let once = reorder_vec_msb(input.clone());
            let twice = reorder_vec_msb(once);
            assert_eq!(twice, input, "mismatch at num_vars={}", num_vars);
        }
    }

    #[test]
    fn msb_num_vars_zero_edge_case() {
        let input = vec![F64::from(42u64)];
        let got = reorder_vec_msb(input.clone());
        assert_eq!(got, input);
    }
}
