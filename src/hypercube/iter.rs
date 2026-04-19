//! Hypercube iterators over `{0,1}^v` in two orders.
//!
//! Each order yields [`HypercubePoint`] structs containing the index and
//! number of variables. Bit access is via [`HypercubePoint::bit(j)`].
//!
//! # Orders
//!
//! - [`Ascending`]: 0, 1, 2, ..., 2^v − 1 (LSB layout). Pairs `(2k, 2k+1)`
//!   differ in the least-significant bit. Use for streaming/blendy provers.
//! - [`BitReverse`]: bit-reversal permutation (MSB layout). Pairs `(k, k+L/2)`
//!   differ in the most-significant bit. Use for in-memory time provers.
//!
//! Both iterators are zero-allocation and `ExactSizeIterator`.

/// A point on the Boolean hypercube `{0,1}^v`.
extern crate alloc;
use alloc::vec::Vec;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HypercubePoint {
    /// The index in the current iteration order.
    pub index: usize,
    /// Number of variables.
    pub num_vars: usize,
}

impl HypercubePoint {
    /// Bit `j` of this point (0 or 1), where bit 0 is the least significant.
    #[inline]
    pub fn bit(&self, j: usize) -> bool {
        debug_assert!(j < self.num_vars);
        (self.index >> j) & 1 == 1
    }

    /// All bits as a `Vec<bool>`, LSB first.
    pub fn bits(&self) -> Vec<bool> {
        (0..self.num_vars).map(|j| self.bit(j)).collect()
    }
}

// ─── Ascending ─────────────────────────────────────────────────────────────

/// Iterate over `{0,1}^v` in ascending order: 0, 1, 2, ..., 2^v − 1.
///
/// This is the LSB (pair-split) layout. Adjacent pairs `(2k, 2k+1)` differ
/// in the least-significant bit. Use for streaming and blendy provers where
/// sequential memory access and cache-friendly prefetching matter.
pub struct Ascending {
    num_vars: usize,
    current: usize,
    end: usize,
}

impl Ascending {
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            current: 0,
            end: 1 << num_vars,
        }
    }
}

impl Iterator for Ascending {
    type Item = HypercubePoint;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }
        let point = HypercubePoint {
            index: self.current,
            num_vars: self.num_vars,
        };
        self.current += 1;
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Ascending {}

// ─── BitReverse ────────────────────────────────────────────────────────────

/// Iterate over `{0,1}^v` in bit-reversal (MSB) order.
///
/// Index `i` maps to `reverse_bits(i) >> (usize::BITS - v)`.
/// One hardware instruction on aarch64 (`RBIT`), fast on x86 too.
///
/// This is the MSB (half-split) layout. Pairs `(k, k + 2^(v-1))` differ
/// in the most-significant bit. Use for in-memory time provers and WHIR.
pub struct BitReverse {
    num_vars: usize,
    shift: u32,
    current: usize,
    end: usize,
}

impl BitReverse {
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            shift: if num_vars == 0 {
                0
            } else {
                usize::BITS - num_vars as u32
            },
            current: 0,
            end: 1 << num_vars,
        }
    }
}

impl Iterator for BitReverse {
    type Item = HypercubePoint;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }
        let reversed = if self.num_vars == 0 {
            0
        } else {
            self.current.reverse_bits() >> self.shift
        };
        let point = HypercubePoint {
            index: reversed,
            num_vars: self.num_vars,
        };
        self.current += 1;
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BitReverse {}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascending_3vars() {
        let points: Vec<usize> = Ascending::new(3).map(|p| p.index).collect();
        assert_eq!(points, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn ascending_exact_size() {
        let iter = Ascending::new(4);
        assert_eq!(iter.len(), 16);
    }

    #[test]
    fn bit_reverse_3vars() {
        let points: Vec<usize> = BitReverse::new(3).map(|p| p.index).collect();
        // 3-bit reversal: 000→000, 001→100, 010→010, 011→110, ...
        assert_eq!(points, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn bit_reverse_is_permutation() {
        let mut points: Vec<usize> = BitReverse::new(4).map(|p| p.index).collect();
        points.sort();
        assert_eq!(points, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn bit_reverse_0vars() {
        let points: Vec<usize> = BitReverse::new(0).map(|p| p.index).collect();
        assert_eq!(points, vec![0]);
    }

    #[test]
    fn bit_reverse_involution() {
        // Applying bit-reversal twice should give back the identity.
        for num_vars in 1..=6 {
            let once: Vec<usize> = BitReverse::new(num_vars).map(|p| p.index).collect();
            let mut twice = vec![0usize; once.len()];
            for (i, &r) in once.iter().enumerate() {
                twice[r] = i;
            }
            // twice[bit_reverse(i)] = i, so twice should be the bit-reverse map again
            let expected: Vec<usize> = BitReverse::new(num_vars).map(|p| p.index).collect();
            assert_eq!(twice, expected, "num_vars={num_vars}");
        }
    }

    #[test]
    fn point_bit_access() {
        let p = HypercubePoint {
            index: 0b101,
            num_vars: 3,
        };
        assert!(p.bit(0)); // LSB
        assert!(!p.bit(1));
        assert!(p.bit(2)); // MSB
        assert_eq!(p.bits(), vec![true, false, true]);
    }
}
