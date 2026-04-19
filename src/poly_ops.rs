//! Zero-allocation polynomial arithmetic on coefficient slices.
//!
//! All functions operate on `&[F]` or `&mut [F]` in ascending degree order
//! (same layout as `DensePolynomial::coeffs`). The caller owns the memory —
//! stack arrays, pre-allocated buffers, or flat fold buffers all work.
//!
//! Designed to eventually upstream into `ark-poly::DensePolynomial` as
//! in-place methods.

use ark_ff::Field;
use ark_poly::univariate::DensePolynomial;

/// Schoolbook polynomial multiplication: `out = a * b`.
///
/// `out` must have length ≥ `a.len() + b.len() - 1`.
/// Zeroes `out` before writing.
///
/// # Panics
///
/// Panics if `out` is too short, or if either input is empty.
#[inline]
pub fn mul_into<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = a.len() + b.len() - 1;
    debug_assert!(
        out.len() >= n,
        "out.len()={} but need {} for deg {} × deg {}",
        out.len(),
        n,
        a.len() - 1,
        b.len() - 1
    );
    for o in out[..n].iter_mut() {
        *o = F::ZERO;
    }
    for (i, &ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
}

/// Fused multiply-accumulate: `out += a * b`.
///
/// `out` must have length ≥ `a.len() + b.len() - 1`.
/// Does NOT zero `out` — accumulates into existing values.
#[inline]
pub fn mul_add_into<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = a.len() + b.len() - 1;
    debug_assert!(out.len() >= n);
    for (i, &ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
}

/// In-place addition: `a += b`.
///
/// `a` must have length ≥ `b.len()`.
#[inline]
pub fn add_assign<F: Field>(a: &mut [F], b: &[F]) {
    debug_assert!(a.len() >= b.len());
    for (ai, &bi) in a.iter_mut().zip(b) {
        *ai += bi;
    }
}

/// In-place subtraction: `a -= b`.
///
/// `a` must have length ≥ `b.len()`.
#[inline]
pub fn sub_assign<F: Field>(a: &mut [F], b: &[F]) {
    debug_assert!(a.len() >= b.len());
    for (ai, &bi) in a.iter_mut().zip(b) {
        *ai -= bi;
    }
}

/// Subtraction into buffer: `out = a - b`.
///
/// `out` must have length ≥ `max(a.len(), b.len())`.
#[inline]
pub fn sub_into<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = a.len().max(b.len());
    debug_assert!(out.len() >= n);
    for i in 0..n {
        let ai = if i < a.len() { a[i] } else { F::ZERO };
        let bi = if i < b.len() { b[i] } else { F::ZERO };
        out[i] = ai - bi;
    }
}

/// Fused scale-and-add: `a += s * b`.
///
/// `a` must have length ≥ `b.len()`.
#[inline]
pub fn add_scaled<F: Field>(a: &mut [F], s: F, b: &[F]) {
    debug_assert!(a.len() >= b.len());
    if s.is_zero() {
        return;
    }
    if s.is_one() {
        add_assign(a, b);
        return;
    }
    for (ai, &bi) in a.iter_mut().zip(b) {
        *ai += s * bi;
    }
}

/// In-place scaling: `a *= s`.
#[inline]
pub fn scale<F: Field>(a: &mut [F], s: F) {
    for ai in a.iter_mut() {
        *ai *= s;
    }
}

/// Evaluate polynomial at `x` via Horner's method.
///
/// `coeffs[0] + coeffs[1]*x + coeffs[2]*x² + ...`
#[inline]
pub fn eval_at<F: Field>(coeffs: &[F], x: F) -> F {
    if coeffs.is_empty() {
        return F::ZERO;
    }
    let mut result = *coeffs.last().unwrap();
    for &c in coeffs.iter().rev().skip(1) {
        result = result * x + c;
    }
    result
}

/// Copy coefficients: `dst[..src.len()] = src`.
#[inline]
pub fn copy_into<F: Field>(dst: &mut [F], src: &[F]) {
    debug_assert!(dst.len() >= src.len());
    dst[..src.len()].copy_from_slice(src);
}

/// Zero a coefficient buffer.
#[inline]
pub fn zero<F: Field>(buf: &mut [F]) {
    for b in buf.iter_mut() {
        *b = F::ZERO;
    }
}

/// Convert a coefficient slice to `DensePolynomial`.
///
/// This is the ONE place that allocates — use at the end when you need
/// to return a `DensePolynomial` to arkworks APIs.
pub fn to_dense_poly<F: Field>(coeffs: &[F]) -> DensePolynomial<F> {
    let mut v = coeffs.to_vec();
    // Trim trailing zeros (DensePolynomial invariant)
    while v.last() == Some(&F::ZERO) && v.len() > 1 {
        v.pop();
    }
    DensePolynomial { coeffs: v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::F64;
    use ark_ff::{UniformRand, Zero};
    use ark_poly::{DenseUVPolynomial, Polynomial};
    use ark_std::{rand::RngCore, test_rng};

    #[test]
    fn test_mul_into_matches_naive_mul() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let deg_a = (rng.next_u32() % 8) as usize;
            let deg_b = (rng.next_u32() % 8) as usize;
            let a: Vec<F64> = (0..=deg_a).map(|_| F64::rand(&mut rng)).collect();
            let b: Vec<F64> = (0..=deg_b).map(|_| F64::rand(&mut rng)).collect();

            let expected = DensePolynomial::from_coefficients_vec(a.clone())
                .naive_mul(&DensePolynomial::from_coefficients_vec(b.clone()));

            let mut out = vec![F64::zero(); a.len() + b.len() - 1];
            mul_into(&mut out, &a, &b);

            for (i, (&e, &o)) in expected.coeffs.iter().zip(out.iter()).enumerate() {
                assert_eq!(e, o, "mul_into mismatch at coeff {i}");
            }
        }
    }

    #[test]
    fn test_mul_add_into_accumulates() {
        let a = [F64::from(1u64), F64::from(2u64)]; // 1 + 2x
        let b = [F64::from(3u64), F64::from(4u64)]; // 3 + 4x
                                                    // a*b = 3 + 10x + 8x²

        let mut out = [F64::from(10u64), F64::zero(), F64::zero()]; // start with 10
        mul_add_into(&mut out, &a, &b);
        // out should be [13, 10, 8]
        assert_eq!(out[0], F64::from(13u64));
        assert_eq!(out[1], F64::from(10u64));
        assert_eq!(out[2], F64::from(8u64));
    }

    #[test]
    fn test_add_scaled() {
        let mut a = [F64::from(1u64), F64::from(2u64), F64::from(3u64)];
        let b = [F64::from(10u64), F64::from(20u64)];
        let s = F64::from(5u64);

        add_scaled(&mut a, s, &b);
        // a = [1+50, 2+100, 3] = [51, 102, 3]
        assert_eq!(a[0], F64::from(51u64));
        assert_eq!(a[1], F64::from(102u64));
        assert_eq!(a[2], F64::from(3u64));
    }

    #[test]
    fn test_eval_at_matches_polynomial() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let deg = (rng.next_u32() % 10) as usize;
            let coeffs: Vec<F64> = (0..=deg).map(|_| F64::rand(&mut rng)).collect();
            let x = F64::rand(&mut rng);

            let expected = DensePolynomial::from_coefficients_vec(coeffs.clone()).evaluate(&x);
            let got = eval_at(&coeffs, x);
            assert_eq!(expected, got);
        }
    }

    #[test]
    fn test_sub_into() {
        let a = [F64::from(10u64), F64::from(20u64), F64::from(30u64)];
        let b = [F64::from(1u64), F64::from(2u64), F64::from(3u64)];
        let mut out = [F64::zero(); 3];
        sub_into(&mut out, &a, &b);
        assert_eq!(out[0], F64::from(9u64));
        assert_eq!(out[1], F64::from(18u64));
        assert_eq!(out[2], F64::from(27u64));
    }

    #[test]
    fn test_to_dense_poly_trims_zeros() {
        let coeffs = [F64::from(1u64), F64::from(2u64), F64::zero(), F64::zero()];
        let p = to_dense_poly(&coeffs);
        assert_eq!(p.coeffs.len(), 2);
        assert_eq!(p.coeffs[0], F64::from(1u64));
        assert_eq!(p.coeffs[1], F64::from(2u64));
    }
}
