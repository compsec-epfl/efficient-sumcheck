//! Zero-allocation dense polynomial arithmetic on coefficient slices.

use crate::field::SumcheckField;

/// Multiply polynomials `a` and `b`, writing the result into `out`.
///
/// `out` must have length `>= a.len() + b.len() - 1`. Existing contents
/// are overwritten. No heap allocation.
///
/// `a = [a_0, a_1, ..., a_m]`, `b = [b_0, b_1, ..., b_n]`.
/// `out[k] = Σ_{i+j=k} a_i · b_j`.
pub fn mul_into<F: SumcheckField>(out: &mut [F], a: &[F], b: &[F]) {
    if a.is_empty() || b.is_empty() {
        for o in out.iter_mut() {
            *o = F::ZERO;
        }
        return;
    }
    let result_len = a.len() + b.len() - 1;
    debug_assert!(out.len() >= result_len);

    for o in out.iter_mut() {
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

/// Add `scalar * p` to `out` in place: `out[i] += scalar · p[i]`.
///
/// If `p` is longer than `out`, the extra terms are ignored.
/// No allocation.
pub fn add_scaled<F: SumcheckField>(out: &mut [F], scalar: F, p: &[F]) {
    let len = out.len().min(p.len());
    for i in 0..len {
        out[i] += scalar * p[i];
    }
}

/// Evaluate polynomial from coefficients at `x` via Horner's method.
///
/// Alias for [`eval_horner`](super::eval_horner).
#[inline]
pub fn eval_at<F: SumcheckField>(coeffs: &[F], x: F) -> F {
    super::eval_horner(coeffs, x)
}

#[cfg(test)]
#[cfg(feature = "arkworks")]
mod tests {
    use super::*;
    use crate::tests::F64;

    #[test]
    fn mul_linear_times_linear() {
        // (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
        let a = [F64::from(1u64), F64::from(2u64)];
        let b = [F64::from(3u64), F64::from(4u64)];
        let mut out = [F64::ZERO; 3];
        mul_into(&mut out, &a, &b);
        assert_eq!(out[0], F64::from(3u64));
        assert_eq!(out[1], F64::from(10u64));
        assert_eq!(out[2], F64::from(8u64));
    }

    #[test]
    fn mul_by_zero() {
        let a = [F64::from(1u64), F64::from(2u64)];
        let b: [F64; 0] = [];
        let mut out = [F64::from(99u64); 3];
        mul_into(&mut out, &a, &b);
        assert!(out.iter().all(|&x| x == F64::ZERO));
    }

    #[test]
    fn add_scaled_basic() {
        let mut out = [F64::from(1u64), F64::from(2u64), F64::from(3u64)];
        let p = [F64::from(10u64), F64::from(20u64)];
        add_scaled(&mut out, F64::from(3u64), &p);
        assert_eq!(out[0], F64::from(31u64)); // 1 + 3*10
        assert_eq!(out[1], F64::from(62u64)); // 2 + 3*20
        assert_eq!(out[2], F64::from(3u64)); // unchanged
    }

    #[test]
    fn eval_at_matches_horner() {
        let coeffs = [F64::from(1u64), F64::from(2u64), F64::from(3u64)];
        let x = F64::from(4u64);
        assert_eq!(eval_at(&coeffs, x), super::super::eval_horner(&coeffs, x));
    }
}
