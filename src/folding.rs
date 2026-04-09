pub mod protogalaxy {
    use ark_ff::{Field, Zero};
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};

    /// Fold `n` polynomials using `log_n` linear coefficient pairs `(a, b)`.
    ///
    /// At each level: `result[i] = p[2i] + (a + b·X)·(p[2i+1] - p[2i])`.
    ///
    /// This version minimizes allocation by working on flat coefficient buffers
    /// and folding in-place. Each polynomial at level `k` has degree ≤ `k`,
    /// so coefficients are stored in fixed-width slots of size `max_degree + 1`.
    pub fn fold<F: Field>(
        coeffs: impl Iterator<Item = (F, F)>,
        polys: Vec<DensePolynomial<F>>,
    ) -> DensePolynomial<F> {
        let coeffs_vec: Vec<(F, F)> = coeffs.collect();
        let n_levels = coeffs_vec.len();

        if polys.is_empty() {
            return DensePolynomial::zero();
        }
        if polys.len() == 1 {
            return polys.into_iter().next().unwrap();
        }

        // Maximum degree after all folds: initial max degree + n_levels
        // (each level multiplies by a degree-1 poly, adding 1 to the degree).
        let init_max_deg = polys.iter().map(|p| p.degree()).max().unwrap_or(0);
        let final_max_deg = init_max_deg + n_levels;
        let slot = final_max_deg + 1; // coefficient slot width

        // Pack all polynomials into a flat buffer with fixed-width slots.
        let mut n_polys = polys.len();
        let mut buf = vec![F::ZERO; n_polys * slot];
        for (i, p) in polys.into_iter().enumerate() {
            for (j, c) in p.coeffs.into_iter().enumerate() {
                buf[i * slot + j] = c;
            }
        }

        // Current degree of polynomials at this level.
        let mut cur_deg = init_max_deg;

        // Scratch buffer for the diff polynomial (reused across levels).
        let mut diff = vec![F::ZERO; slot];

        for (level, &(a, b)) in coeffs_vec.iter().enumerate() {
            let _ = level;
            let half = n_polys / 2;

            for i in 0..half {
                let p0_off = (2 * i) * slot;
                let p1_off = (2 * i + 1) * slot;
                let out_off = i * slot;

                // diff = p1 - p0 (degree ≤ cur_deg)
                for j in 0..=cur_deg {
                    diff[j] = buf[p1_off + j] - buf[p0_off + j];
                }

                // result = p0 + (a + b·X) · diff
                //        = p0 + a·diff + b·X·diff
                //        = p0[j] + a·diff[j] + b·diff[j-1]  for each j
                //
                // New degree = cur_deg + 1

                // Compute in-place into buf[out_off..].
                // Process from high to low to avoid overwriting p0 before reading it
                // (out_off ≤ p0_off since i ≤ 2i, and slots don't overlap after halving).

                // Highest coefficient (j = cur_deg + 1): only b·diff[cur_deg]
                buf[out_off + cur_deg + 1] = b * diff[cur_deg];

                // Middle coefficients (j = cur_deg down to 1): p0[j] + a·diff[j] + b·diff[j-1]
                for j in (1..=cur_deg).rev() {
                    buf[out_off + j] = buf[p0_off + j] + a * diff[j] + b * diff[j - 1];
                }

                // Lowest coefficient (j = 0): p0[0] + a·diff[0]
                buf[out_off] = buf[p0_off] + a * diff[0];

                // Zero out remaining slots
                for j in (cur_deg + 2)..slot {
                    buf[out_off + j] = F::ZERO;
                }
            }

            cur_deg += 1;
            n_polys = half;
        }

        // Extract the single remaining polynomial from slot 0.
        debug_assert_eq!(n_polys, 1);
        let final_deg = cur_deg.min(final_max_deg);
        let mut result_coeffs: Vec<F> = buf[..=final_deg].to_vec();

        // Trim trailing zeros
        while result_coeffs.last() == Some(&F::ZERO) && result_coeffs.len() > 1 {
            result_coeffs.pop();
        }

        DensePolynomial::from_coefficients_vec(result_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::protogalaxy::fold;
    use crate::tests::F64;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};

    #[test]
    fn test_fold_two_polys() {
        // p0 = 1, p1 = X, (a, b) = (3, 5)
        // result = (1 - a) + (a - b)X + bX^2
        let p0 = DensePolynomial::from_coefficients_vec(vec![F64::from(1u64)]);
        let p1 = DensePolynomial::from_coefficients_vec(vec![F64::from(0u64), F64::from(1u64)]);

        let a = F64::from(3u64);
        let b = F64::from(5u64);
        let result = fold(vec![(a, b)].into_iter(), vec![p0, p1]);

        assert_eq!(result.coeffs.len(), 3);
        assert_eq!(result.coeffs[0], F64::from(1u64) - a);
        assert_eq!(result.coeffs[1], a - b);
        assert_eq!(result.coeffs[2], b);
    }

    #[test]
    fn test_fold_four_polys() {
        // 4 constant polys [1, 2, 3, 4], coeffs = (1, 0) at each level
        // (a + b·X) = 1, so fold selects p[1] each time: [1,2,3,4] → [2,4] → [4]
        let polys: Vec<DensePolynomial<F64>> = (1..=4u64)
            .map(|c| DensePolynomial::from_coefficients_vec(vec![F64::from(c)]))
            .collect();

        let coeffs = vec![(F64::from(1u64), F64::from(0u64)); 2];
        let result = fold(coeffs.into_iter(), polys);

        assert_eq!(result.coeffs.len(), 1);
        assert_eq!(result.coeffs[0], F64::from(4u64));
    }

    #[test]
    fn test_fold_matches_naive() {
        // Compare optimized fold against a naive reference for random inputs.
        use ark_ff::UniformRand;
        use ark_std::test_rng;

        let mut rng = test_rng();

        // 8 random degree-2 polynomials, 3 fold levels
        let polys: Vec<DensePolynomial<F64>> = (0..8)
            .map(|_| {
                DensePolynomial::from_coefficients_vec(vec![
                    F64::rand(&mut rng),
                    F64::rand(&mut rng),
                    F64::rand(&mut rng),
                ])
            })
            .collect();

        let coeffs: Vec<(F64, F64)> = (0..3)
            .map(|_| (F64::rand(&mut rng), F64::rand(&mut rng)))
            .collect();

        // Naive fold (original algorithm)
        let naive_result = {
            let mut ps = polys.clone();
            for &(a, b) in &coeffs {
                ps = ps
                    .chunks(2)
                    .map(|p| {
                        &p[0]
                            + DensePolynomial::from_coefficients_vec(vec![a, b])
                                .naive_mul(&(&p[1] - &p[0]))
                    })
                    .collect();
            }
            ps.pop().unwrap()
        };

        // Optimized fold
        let opt_result = fold(coeffs.into_iter(), polys);

        assert_eq!(naive_result.coeffs.len(), opt_result.coeffs.len());
        for (n, o) in naive_result.coeffs.iter().zip(opt_result.coeffs.iter()) {
            assert_eq!(*n, *o, "coefficient mismatch");
        }
    }
}
