pub mod protogalaxy {
    use ark_ff::Field;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    /// Fold `n` polynomials using `log_n` linear coefficient pairs `(a, b)`.
    ///
    /// At each level: `p[0] + (a + b·X)·(p[1] - p[0])`.
    pub fn fold<F: Field>(
        coeffs: impl Iterator<Item = (F, F)>,
        mut polys: Vec<DensePolynomial<F>>,
    ) -> DensePolynomial<F> {
        for (a, b) in coeffs {
            #[cfg(feature = "parallel")]
            {
                polys = polys
                    .par_chunks(2)
                    .map(|p| {
                        &p[0]
                            + DensePolynomial::from_coefficients_vec(vec![a, b])
                                .naive_mul(&(&p[1] - &p[0]))
                    })
                    .collect();
            }
            #[cfg(not(feature = "parallel"))]
            {
                polys = polys
                    .chunks(2)
                    .map(|p| {
                        &p[0]
                            + DensePolynomial::from_coefficients_vec(vec![a, b])
                                .naive_mul(&(&p[1] - &p[0]))
                    })
                    .collect();
            }
        }
        assert_eq!(polys.len(), 1);
        polys.pop().unwrap()
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
}
