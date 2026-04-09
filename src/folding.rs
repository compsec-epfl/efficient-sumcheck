pub mod protogalaxy {
    use ark_ff::{Field, Zero};
    use ark_poly::{univariate::DensePolynomial, Polynomial};

    use crate::poly_ops;

    /// Fold `n` polynomials using `log_n` linear coefficient pairs `(a, b)`.
    ///
    /// At each level: `result[i] = p[2i] + (a + b·X)·(p[2i+1] - p[2i])`.
    ///
    /// Uses [`poly_ops`] for zero-allocation arithmetic on flat coefficient buffers.
    /// Each polynomial at level `k` has degree ≤ initial_degree + `k`,
    /// stored in fixed-width slots.
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

        let init_max_deg = polys.iter().map(|p| p.degree()).max().unwrap_or(0);
        let final_max_deg = init_max_deg + n_levels;
        let slot = final_max_deg + 1;

        // Pack into flat buffer with fixed-width slots.
        let mut n_polys = polys.len();
        let mut buf = vec![F::ZERO; n_polys * slot];
        for (i, p) in polys.into_iter().enumerate() {
            poly_ops::copy_into(&mut buf[i * slot..], &p.coeffs);
        }

        let mut cur_deg = init_max_deg;
        let mut diff = vec![F::ZERO; slot];

        for &(a, b) in &coeffs_vec {
            let half = n_polys / 2;

            for i in 0..half {
                let p0_off = (2 * i) * slot;
                let p1_off = (2 * i + 1) * slot;
                let out_off = i * slot;
                let deg = cur_deg + 1; // new degree after this level

                // diff[0..=cur_deg] = p1 - p0
                poly_ops::sub_into(
                    &mut diff[..=cur_deg],
                    &buf[p1_off..p1_off + cur_deg + 1],
                    &buf[p0_off..p0_off + cur_deg + 1],
                );

                // result = p0 + a·diff + b·X·diff
                // Process high-to-low to allow in-place when out_off ≤ p0_off.
                buf[out_off + deg] = b * diff[cur_deg];
                for j in (1..=cur_deg).rev() {
                    buf[out_off + j] = buf[p0_off + j] + a * diff[j] + b * diff[j - 1];
                }
                buf[out_off] = buf[p0_off] + a * diff[0];

                poly_ops::zero(&mut buf[out_off + deg + 1..out_off + slot]);
            }

            cur_deg += 1;
            n_polys = half;
        }

        debug_assert_eq!(n_polys, 1);
        poly_ops::to_dense_poly(&buf[..=cur_deg.min(final_max_deg)])
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
