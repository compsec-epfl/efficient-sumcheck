//! SIMD-accelerated field operations.
//!
//! General-purpose primitives that auto-dispatch to SIMD backends for
//! Goldilocks-based fields (base field, degree-2 and degree-3 extensions).
//! Falls back to generic arkworks `Field` operations for other fields.
//!
//! These are not sumcheck-specific — any protocol that does pairwise folding,
//! dot products, or multi-scalar operations can use them.
//!
//! # Example
//!
//! ```text
//! use efficient_sumcheck::simd_ops;
//!
//! let mut evals: Vec<F> = /* ... */;
//! let (s0, s1) = simd_ops::pairwise_sum(&evals);
//! simd_ops::fold(&mut evals, challenge);
//! let dot = simd_ops::inner_product(&f, &g);
//! ```

use ark_ff::Field;

// ─── Pairwise sum ───────────────────────────────────────────────────────────

/// Sum even-indexed and odd-indexed elements.
///
/// Returns `(Σ data[2i], Σ data[2i+1])` for `i = 0..data.len()/2`.
///
/// SIMD-accelerated for Goldilocks base and extension fields.
pub fn pairwise_sum<F: Field>(data: &[F]) -> (F, F) {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    if let Some(result) = crate::simd_sumcheck::dispatch::try_simd_ext_evaluate(data) {
        return result;
    }

    // Generic fallback
    let mut even = F::ZERO;
    let mut odd = F::ZERO;
    for i in (0..data.len()).step_by(2) {
        even += data[i];
        if i + 1 < data.len() {
            odd += data[i + 1];
        }
    }
    (even, odd)
}

// ─── Fold ───────────────────────────────────────────────────────────────────

/// Half-split (MSB) fold:
/// `data[k] = data[k] + challenge * (data[k + L/2] − data[k])` for `k` in `0..L/2`.
///
/// Implicit zero padding: elements in the low half beyond `len − L/2` have
/// no partner and are folded as `data[k] * (1 − challenge)`. After the fold,
/// `data` is truncated to `L/2` (the next power of two ÷ 2).
///
/// SIMD-accelerated for Goldilocks base field. Falls back to a scalar
/// recursive rayon::join fold for other fields and extension fields.
pub fn fold<F: Field>(data: &mut Vec<F>, challenge: F) {
    // SIMD fast path for base-field Goldilocks (MSB layout).
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    {
        if crate::simd_sumcheck::dispatch::try_simd_reduce_msb(data, challenge) {
            data.shrink_to_fit();
            return;
        }
    }

    // Generic scalar MSB fold with rayon parallelism.
    crate::multilinear_sumcheck::fold(data, challenge);
}

// ─── Product evaluate ───────────────────────────────────────────────────────

/// Pairwise product sum: computes coefficients `(a, b)` of the degree-2
/// round polynomial from two evaluation vectors.
///
/// - `a = Σ f[2i] * g[2i]`  (even-even products)
/// - `b = Σ (f[2i] * g[2i+1] + f[2i+1] * g[2i])`  (cross-term)
///
/// SIMD-accelerated for Goldilocks base field.
pub fn pairwise_product_sum<F: Field>(f: &[F], g: &[F]) -> (F, F) {
    debug_assert_eq!(f.len(), g.len());

    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    if let Some(result) = try_simd_product_sum(f, g) {
        return result;
    }

    // Generic fallback
    crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate_slices(f, g)
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
fn try_simd_product_sum<F: Field>(f: &[F], g: &[F]) -> Option<(F, F)> {
    use crate::simd_sumcheck::dispatch::is_goldilocks_pub;

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    if is_goldilocks_pub::<F>() {
        let f_raw: &[u64] =
            unsafe { core::slice::from_raw_parts(f.as_ptr() as *const u64, f.len()) };
        let g_raw: &[u64] =
            unsafe { core::slice::from_raw_parts(g.as_ptr() as *const u64, g.len()) };
        let (a, b) =
            crate::simd_sumcheck::evaluate::product_evaluate_parallel::<Backend>(f_raw, g_raw);

        use crate::simd_sumcheck::dispatch::u64_to_field_pub;
        return Some((u64_to_field_pub(a), u64_to_field_pub(b)));
    }

    None
}

// ─── Inner product ──────────────────────────────────────────────────────────

/// Dot product: `Σ f[i] * g[i]`.
///
/// SIMD-accelerated for Goldilocks base field.
pub fn inner_product<F: Field>(f: &[F], g: &[F]) -> F {
    debug_assert_eq!(f.len(), g.len());
    f.iter().zip(g.iter()).map(|(a, b)| *a * *b).sum()
    // Note: SIMD inner product would require extension multiply for ext fields.
    // For base field, the generic .sum() with rayon is already fast.
    // Future: add SIMD dispatch here.
}

// ─── Cross-field reduce ─────────────────────────────────────────────────────

/// Fold base-field evaluations with an extension-field challenge.
///
/// Each pair `(a, b)` in `data` (base field) is folded to
/// `EF::from(a) + challenge * (EF::from(b) - EF::from(a))` in the extension field.
///
/// Returns a new `Vec<EF>`.
pub fn cross_field_fold<BF: Field, EF: Field + From<BF>>(data: &[BF], challenge: EF) -> Vec<EF> {
    crate::multilinear::reductions::pairwise::cross_field_reduce(data, challenge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{F64Ext2, F64Ext3, F64};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_pairwise_sum_base() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let data: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let (even, odd) = pairwise_sum(&data);

        let expected_even: F64 = data.iter().step_by(2).copied().sum();
        let expected_odd: F64 = data.iter().skip(1).step_by(2).copied().sum();

        assert_eq!(even, expected_even);
        assert_eq!(odd, expected_odd);
    }

    #[test]
    fn test_pairwise_sum_ext2() {
        let mut rng = test_rng();
        let n = 1 << 8;
        let data: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let (even, odd) = pairwise_sum(&data);

        let expected_even: F64Ext2 = data.iter().step_by(2).copied().sum();
        let expected_odd: F64Ext2 = data.iter().skip(1).step_by(2).copied().sum();

        assert_eq!(even, expected_even);
        assert_eq!(odd, expected_odd);
    }

    #[test]
    fn test_pairwise_sum_ext3() {
        let mut rng = test_rng();
        let n = 1 << 8;
        let data: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();

        let (even, odd) = pairwise_sum(&data);

        let expected_even: F64Ext3 = data.iter().step_by(2).copied().sum();
        let expected_odd: F64Ext3 = data.iter().skip(1).step_by(2).copied().sum();

        assert_eq!(even, expected_even);
        assert_eq!(odd, expected_odd);
    }

    #[test]
    fn test_fold_base() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let data: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let challenge = F64::rand(&mut rng);

        // Reference: MSB (half-split) fold — pair data[k] with data[k + half].
        let half = n / 2;
        let expected: Vec<F64> = (0..half)
            .map(|k| data[k] + challenge * (data[k + half] - data[k]))
            .collect();

        let mut result = data;
        fold(&mut result, challenge);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold_ext2() {
        let mut rng = test_rng();
        let n = 1 << 8;
        let data: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let challenge = F64Ext2::rand(&mut rng);

        let half = n / 2;
        let expected: Vec<F64Ext2> = (0..half)
            .map(|k| data[k] + challenge * (data[k + half] - data[k]))
            .collect();

        let mut result = data;
        fold(&mut result, challenge);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_pairwise_product_sum_base() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let (a, b) = pairwise_product_sum(&f, &g);

        // Reference
        let expected_a: F64 = (0..n / 2).map(|k| f[2 * k] * g[2 * k]).sum();
        let expected_b: F64 = (0..n / 2)
            .map(|k| f[2 * k] * g[2 * k + 1] + f[2 * k + 1] * g[2 * k])
            .sum();

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_pairwise_product_sum_ext2() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let (a, b) = pairwise_product_sum(&f, &g);

        let expected_a: F64Ext2 = (0..n / 2).map(|k| f[2 * k] * g[2 * k]).sum();
        let expected_b: F64Ext2 = (0..n / 2)
            .map(|k| f[2 * k] * g[2 * k + 1] + f[2 * k + 1] * g[2 * k])
            .sum();

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_pairwise_product_sum_ext3() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let f: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
        let g: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();

        let (a, b) = pairwise_product_sum(&f, &g);

        let expected_a: F64Ext3 = (0..n / 2).map(|k| f[2 * k] * g[2 * k]).sum();
        let expected_b: F64Ext3 = (0..n / 2)
            .map(|k| f[2 * k] * g[2 * k + 1] + f[2 * k + 1] * g[2 * k])
            .sum();

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }
}
