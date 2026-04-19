//! Polynomial evaluation: Horner's method and barycentric Lagrange interpolation.

use crate::field::SumcheckField;

/// Evaluate a polynomial from its coefficients at point `x` via Horner's method.
///
/// `coeffs = [c_0, c_1, ..., c_d]` represents `p(X) = c_0 + c_1·X + ... + c_d·X^d`.
///
/// Cost: `d` multiplications + `d` additions. Zero allocation.
#[inline]
pub fn eval_horner<F: SumcheckField>(coeffs: &[F], x: F) -> F {
    if coeffs.is_empty() {
        return F::ZERO;
    }
    let mut result = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        result = result * x + coeffs[i];
    }
    result
}

/// Evaluate a polynomial from its evaluations at `{0, 1, ..., d}` at
/// an arbitrary point `x` via barycentric Lagrange interpolation.
///
/// `evals = [p(0), p(1), ..., p(d)]`.
///
/// Cost: O(d) with precomputed [`BarycentricWeights`], O(d²) without.
/// For repeated evaluations at the same degree, precompute weights once.
pub fn eval_from_evals<F: SumcheckField>(evals: &[F], x: F) -> F {
    let d = evals.len();
    if d == 0 {
        return F::ZERO;
    }
    if d == 1 {
        return evals[0];
    }
    if d == 2 {
        // Linear: p(x) = p(0) + x·(p(1) − p(0)).
        return evals[0] + x * (evals[1] - evals[0]);
    }
    BarycentricWeights::new(d - 1).eval(evals, x)
}

/// Precomputed barycentric weights for Lagrange interpolation at `{0, 1, ..., d}`.
///
/// Compute once per degree, reuse across rounds. The verifier calls this
/// once and evaluates O(d) per round instead of O(d²).
///
/// Weight `w_i = 1 / Π_{j≠i} (i − j)` for `i, j ∈ {0, ..., d}`.
/// For consecutive integer nodes these are `(-1)^{d-i} / (i! · (d-i)!)`.
pub struct BarycentricWeights<F: SumcheckField> {
    /// Precomputed `w_i` for each node `i ∈ {0, ..., d}`.
    weights: Vec<F>,
}

impl<F: SumcheckField> BarycentricWeights<F> {
    /// Precompute weights for interpolation at `{0, 1, ..., degree}`.
    pub fn new(degree: usize) -> Self {
        let d = degree + 1; // number of nodes
        let mut weights = Vec::with_capacity(d);
        for i in 0..d {
            let mut w = F::ONE;
            for j in 0..d {
                if j != i {
                    let diff = i as i64 - j as i64;
                    if diff > 0 {
                        w *= F::from_u64(diff as u64);
                    } else {
                        w *= -F::from_u64((-diff) as u64);
                    }
                }
            }
            // w_i = 1 / Π_{j≠i} (i - j)
            weights.push(w.inverse().unwrap_or(F::ZERO));
        }
        Self { weights }
    }

    /// Number of interpolation nodes (degree + 1).
    pub fn num_nodes(&self) -> usize {
        self.weights.len()
    }

    /// Evaluate the interpolated polynomial at `x`.
    ///
    /// `evals` must have length `num_nodes()`.
    ///
    /// Uses the "first form" of the barycentric formula:
    /// `p(x) = Σ_i w_i · L(x) / (x - i) · f(i)`
    /// where `L(x) = Π_j (x - j)`.
    ///
    /// Cost: O(d) multiplications + O(d) additions.
    pub fn eval(&self, evals: &[F], x: F) -> F {
        let d = self.weights.len();
        debug_assert_eq!(evals.len(), d);

        // Check if x is one of the nodes (avoid division by zero).
        for (i, &eval) in evals.iter().enumerate() {
            let node = F::from_u64(i as u64);
            if x == node {
                return eval;
            }
        }

        // Compute (x - 0)(x - 1)...(x - d+1) via prefix/suffix products.
        let x_minus: Vec<F> = (0..d).map(|j| x - F::from_u64(j as u64)).collect();

        let mut prefix = vec![F::ONE; d + 1];
        for i in 0..d {
            prefix[i + 1] = prefix[i] * x_minus[i];
        }
        let mut suffix = vec![F::ONE; d + 1];
        for i in (0..d).rev() {
            suffix[i] = suffix[i + 1] * x_minus[i];
        }

        let mut result = F::ZERO;
        for i in 0..d {
            // numerator = Π_{j≠i} (x - j) = prefix[i] · suffix[i+1]
            let numerator = prefix[i] * suffix[i + 1];
            result += evals[i] * numerator * self.weights[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple field-like wrapper for f64 won't work with SumcheckField
    // (needs Copy + all the ops). Tests use the arkworks F64 type.
    #[cfg(feature = "arkworks")]
    mod ark_tests {
        use super::*;
        use crate::tests::F64;

        #[test]
        fn horner_constant() {
            let coeffs = [F64::from(7u64)];
            assert_eq!(eval_horner(&coeffs, F64::from(42u64)), F64::from(7u64));
        }

        #[test]
        fn horner_linear() {
            // p(x) = 3 + 5x
            let coeffs = [F64::from(3u64), F64::from(5u64)];
            // p(2) = 3 + 10 = 13
            assert_eq!(eval_horner(&coeffs, F64::from(2u64)), F64::from(13u64));
        }

        #[test]
        fn horner_quadratic() {
            // p(x) = 1 + 2x + 3x²
            let coeffs = [F64::from(1u64), F64::from(2u64), F64::from(3u64)];
            // p(4) = 1 + 8 + 48 = 57
            assert_eq!(eval_horner(&coeffs, F64::from(4u64)), F64::from(57u64));
        }

        #[test]
        fn horner_empty() {
            let coeffs: [F64; 0] = [];
            assert_eq!(eval_horner(&coeffs, F64::from(5u64)), F64::ZERO);
        }

        #[test]
        fn barycentric_matches_horner() {
            // p(x) = 1 + 2x + 3x²
            // p(0) = 1, p(1) = 6, p(2) = 17
            let evals = [F64::from(1u64), F64::from(6u64), F64::from(17u64)];
            let coeffs = [F64::from(1u64), F64::from(2u64), F64::from(3u64)];

            // Evaluate at several points and compare.
            for x_val in [0u64, 1, 2, 3, 5, 10, 100] {
                let x = F64::from(x_val);
                let from_coeffs = eval_horner(&coeffs, x);
                let from_evals = eval_from_evals(&evals, x);
                assert_eq!(
                    from_coeffs, from_evals,
                    "mismatch at x={x_val}"
                );
            }
        }

        #[test]
        fn barycentric_linear() {
            // p(0) = 3, p(1) = 7 → p(x) = 3 + 4x → p(5) = 23
            let evals = [F64::from(3u64), F64::from(7u64)];
            assert_eq!(eval_from_evals(&evals, F64::from(5u64)), F64::from(23u64));
        }

        #[test]
        fn barycentric_at_nodes() {
            let evals = [F64::from(10u64), F64::from(20u64), F64::from(30u64)];
            assert_eq!(eval_from_evals(&evals, F64::from(0u64)), F64::from(10u64));
            assert_eq!(eval_from_evals(&evals, F64::from(1u64)), F64::from(20u64));
            assert_eq!(eval_from_evals(&evals, F64::from(2u64)), F64::from(30u64));
        }

        #[test]
        fn precomputed_weights_reuse() {
            let evals = [F64::from(1u64), F64::from(6u64), F64::from(17u64)];
            let weights = BarycentricWeights::new(2);

            // Reuse weights for multiple evaluations.
            let v3 = weights.eval(&evals, F64::from(3u64));
            let v5 = weights.eval(&evals, F64::from(5u64));

            let coeffs = [F64::from(1u64), F64::from(2u64), F64::from(3u64)];
            assert_eq!(v3, eval_horner(&coeffs, F64::from(3u64)));
            assert_eq!(v5, eval_horner(&coeffs, F64::from(5u64)));
        }
    }
}
