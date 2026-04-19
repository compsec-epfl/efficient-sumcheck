//! Sequential Lagrange polynomial over the Boolean hypercube.
//!
//! Maintains `eq(r, x) = Π_j (r_j · x_j + (1 − r_j) · (1 − x_j))`
//! incrementally as you iterate over `{0,1}^v`.
//!
//! Designed to compose with [`Ascending`](crate::hypercube::Ascending):
//! call [`advance_to`](SequentialLagrange::advance_to) with each successive
//! index. The XOR of consecutive indices tells us which bits flipped;
//! each flipped bit requires one multiply+divide to update the product.
//!
//! For ascending order, bit 0 flips every step, bit 1 every 2 steps, etc.
//! The amortized cost per step is O(1) (geometric series: 1 + 1/2 + 1/4 + ... = 2).

extern crate alloc;
use alloc::vec::Vec;
use crate::field::SumcheckField;

/// Sequential Lagrange polynomial `eq(r, ·)` with incremental updates.
///
/// # Usage
///
/// ```ignore
/// use effsc::polynomial::SequentialLagrange;
/// use effsc::hypercube::Ascending;
///
/// let point = vec![r0, r1, r2];
/// let mut lag = SequentialLagrange::new(&point);
///
/// for p in Ascending::new(3) {
///     lag.advance_to(p.index);
///     let eq_val = lag.value();
///     // use eq_val...
/// }
/// ```
pub struct SequentialLagrange<F: SumcheckField> {
    /// Precomputed factors: `factor_one[j] = r_j`, `factor_zero[j] = 1 − r_j`.
    factor_one: Vec<F>,
    factor_zero: Vec<F>,
    /// Current product value `Π_j factor(x_j)`.
    current_value: F,
    /// Current hypercube index (which bits are set).
    current_index: usize,
    /// Number of variables.
    num_vars: usize,
}

impl<F: SumcheckField> SequentialLagrange<F> {
    /// Initialize at the origin (index 0): `eq(r, 0) = Π_j (1 − r_j)`.
    pub fn new(point: &[F]) -> Self {
        let num_vars = point.len();
        let factor_one: Vec<F> = point.to_vec();
        let factor_zero: Vec<F> = point.iter().map(|&r| F::ONE - r).collect();

        // Initial value: eq(r, 0...0) = Π_j (1 - r_j)
        let current_value = factor_zero.iter().copied().fold(F::ONE, |acc, f| acc * f);

        Self {
            factor_one,
            factor_zero,
            current_value,
            current_index: 0,
            num_vars,
        }
    }

    /// Current value of `eq(r, x)` at the current hypercube point.
    #[inline]
    pub fn value(&self) -> F {
        self.current_value
    }

    /// Current hypercube index.
    #[inline]
    pub fn index(&self) -> usize {
        self.current_index
    }

    /// Advance to a new hypercube index.
    ///
    /// Updates the product by flipping the factors for each bit that
    /// changed between the current index and `next_index`.
    ///
    /// For ascending iteration (0, 1, 2, ...), this is amortized O(1):
    /// bit 0 flips every step (cost 1), bit 1 every 2 steps (cost 1/2),
    /// bit 2 every 4 steps (cost 1/4), etc. Total: Σ 1/2^k = 2.
    pub fn advance_to(&mut self, next_index: usize) {
        let diff = self.current_index ^ next_index;
        let mut bits = diff;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            debug_assert!(j < self.num_vars);

            // Determine if bit j flipped 0→1 or 1→0.
            let was_one = (self.current_index >> j) & 1 == 1;
            if was_one {
                // 1→0: replace factor_one[j] with factor_zero[j]
                // new = old / factor_one[j] * factor_zero[j]
                if let Some(inv) = self.factor_one[j].inverse() {
                    self.current_value *= self.factor_zero[j] * inv;
                }
            } else {
                // 0→1: replace factor_zero[j] with factor_one[j]
                // new = old / factor_zero[j] * factor_one[j]
                if let Some(inv) = self.factor_zero[j].inverse() {
                    self.current_value *= self.factor_one[j] * inv;
                }
            }

            bits &= bits - 1; // clear lowest set bit
        }
        self.current_index = next_index;
    }

    /// Reset to index 0.
    pub fn reset(&mut self) {
        self.current_value = self.factor_zero.iter().copied().fold(F::ONE, |acc, f| acc * f);
        self.current_index = 0;
    }
}

#[cfg(test)]
#[cfg(feature = "arkworks")]
mod tests {
    use super::*;
    use crate::tests::F64;

    /// Compute eq(r, x) directly for reference.
    fn eq_direct(point: &[F64], index: usize) -> F64 {
        let num_vars = point.len();
        (0..num_vars).fold(F64::from(1u64), |acc, j| {
            let bit = F64::from(((index >> j) & 1) as u64);
            acc * (point[j] * bit + (F64::from(1u64) - point[j]) * (F64::from(1u64) - bit))
        })
    }

    #[test]
    fn sequential_ascending_matches_direct() {
        use ark_ff::UniformRand;
        use ark_std::rand::{rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let num_vars = 4;
        let point: Vec<F64> = (0..num_vars).map(|_| F64::rand(&mut rng)).collect();

        let mut lag = SequentialLagrange::new(&point);
        assert_eq!(lag.value(), eq_direct(&point, 0));

        for i in 1..(1 << num_vars) {
            lag.advance_to(i);
            let expected = eq_direct(&point, i);
            assert_eq!(
                lag.value(),
                expected,
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn sequential_random_access() {
        use ark_ff::UniformRand;
        use ark_std::rand::{rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(99);
        let num_vars = 3;
        let point: Vec<F64> = (0..num_vars).map(|_| F64::rand(&mut rng)).collect();

        let mut lag = SequentialLagrange::new(&point);

        // Jump around non-sequentially.
        for &idx in &[5, 3, 7, 0, 6, 1, 4, 2] {
            lag.advance_to(idx);
            assert_eq!(lag.value(), eq_direct(&point, idx), "at index {idx}");
        }
    }

    #[test]
    fn sequential_reset() {
        use ark_ff::UniformRand;
        use ark_std::rand::{rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(7);
        let point: Vec<F64> = (0..3).map(|_| F64::rand(&mut rng)).collect();

        let mut lag = SequentialLagrange::new(&point);
        lag.advance_to(5);
        lag.reset();
        assert_eq!(lag.value(), eq_direct(&point, 0));
        assert_eq!(lag.index(), 0);
    }

    #[test]
    fn sequential_composes_with_ascending() {
        use crate::hypercube::Ascending;
        use ark_ff::UniformRand;
        use ark_std::rand::{rngs::StdRng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(123);
        let num_vars = 5;
        let point: Vec<F64> = (0..num_vars).map(|_| F64::rand(&mut rng)).collect();

        let mut lag = SequentialLagrange::new(&point);
        for p in Ascending::new(num_vars) {
            lag.advance_to(p.index);
            assert_eq!(lag.value(), eq_direct(&point, p.index));
        }
    }
}
