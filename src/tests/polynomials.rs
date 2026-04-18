use ark_ff::Field;
use ark_poly::{
    multivariate::{self, SparsePolynomial, SparseTerm, Term},
    DenseMVPolynomial,
};

/*
 * Small polynomials for sanity checking.
 */

pub fn three_variable_polynomial<F: Field>() -> SparsePolynomial<F, SparseTerm> {
    // 4*x_1*x_2 + 7*x_2*x_3 + 2*x_1 + 13*x_2
    multivariate::SparsePolynomial::from_coefficients_slice(
        3,
        &[
            (
                F::from(4_u32),
                multivariate::SparseTerm::new(vec![(0, 1), (1, 1)]),
            ),
            (
                F::from(7_u32),
                multivariate::SparseTerm::new(vec![(1, 1), (2, 1)]),
            ),
            (F::from(2_u32), multivariate::SparseTerm::new(vec![(0, 1)])),
            (F::from(13_u32), multivariate::SparseTerm::new(vec![(1, 1)])),
        ],
    )
}

pub fn three_variable_polynomial_evaluations<F: Field>() -> Vec<F> {
    three_variable_polynomial().to_evaluations()
}

pub fn four_variable_polynomial<F: Field>() -> SparsePolynomial<F, SparseTerm> {
    // 4*x_1*x_2 + 7*x_2*x_3 + 2*x_1 + 13*x_2 + 1x_4
    multivariate::SparsePolynomial::from_coefficients_slice(
        4,
        &[
            (
                F::from(4_u32),
                multivariate::SparseTerm::new(vec![(0, 1), (1, 1)]),
            ),
            (
                F::from(7_u32),
                multivariate::SparseTerm::new(vec![(1, 1), (2, 1)]),
            ),
            (F::from(2_u32), multivariate::SparseTerm::new(vec![(0, 1)])),
            (F::from(13_u32), multivariate::SparseTerm::new(vec![(1, 1)])),
            (F::from(1_u32), multivariate::SparseTerm::new(vec![(3, 1)])),
        ],
    )
}

pub fn four_variable_polynomial_evaluations<F: Field>() -> Vec<F> {
    four_variable_polynomial().to_evaluations()
}

/*
 * Extension trait to evaluate multivariate sparse polynomials on the
 * Boolean hypercube and convert between evaluation and coefficient form.
 */

pub trait Polynomial<F: Field> {
    fn evaluate(&self, point: Vec<F>) -> Option<F>;
    fn to_evaluations(&self) -> Vec<F>;
    fn from_hypercube_evaluations(evaluations: Vec<F>) -> SparsePolynomial<F, SparseTerm>;
}

impl<F: Field> Polynomial<F> for SparsePolynomial<F, SparseTerm> {
    fn evaluate(&self, point: Vec<F>) -> Option<F> {
        assert_eq!(DenseMVPolynomial::<F>::num_vars(self), point.len());
        let mut result = F::ZERO;
        for (coefficient, term) in self.terms().iter() {
            result += term.evaluate(&point) * coefficient;
        }
        Some(result)
    }

    fn to_evaluations(&self) -> Vec<F> {
        let num_vars = DenseMVPolynomial::<F>::num_vars(self);
        let total_points = 1usize << num_vars;
        let mut evaluations = Vec::with_capacity(total_points);

        for index in 0..total_points {
            // Convert index bits to field elements.
            let point: Vec<F> = (0..num_vars)
                .map(|j| {
                    if index >> j & 1 == 1 {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                })
                .collect();
            let mut val = F::ZERO;
            for (coefficient, term) in self.terms().iter() {
                val += term.evaluate(&point) * coefficient;
            }
            evaluations.push(val);
        }

        evaluations
    }

    fn from_hypercube_evaluations(mut evaluations: Vec<F>) -> SparsePolynomial<F, SparseTerm> {
        assert!(
            evaluations.len().is_power_of_two(),
            "evaluations len must be a power of two"
        );
        let num_vars: usize = evaluations.len().ilog2() as usize;
        let n = evaluations.len();

        // Evaluations are in ascending (standard binary) order — no reorder needed.
        // In-place Mobius inversion.
        for i in 0..num_vars {
            for mask in 0..n {
                if mask & (1 << i) != 0 {
                    evaluations[mask] = evaluations[mask] - evaluations[mask ^ (1 << i)];
                }
            }
        }

        // Build sparse polynomial from nonzero coefficients.
        let mut terms = Vec::new();
        for (mask, evaluation) in evaluations.iter().enumerate() {
            if evaluations[mask] != F::zero() {
                let mut exponents = Vec::new();
                for var in 0..num_vars {
                    if mask & (1 << var) != 0 {
                        exponents.push((var, 1));
                    }
                }
                let term = SparseTerm::new(exponents);
                terms.push((*evaluation, term));
            }
        }

        SparsePolynomial::from_coefficients_slice(num_vars, &terms)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        streams::Stream,
        tests::{
            polynomials::{four_variable_polynomial, Polynomial},
            BenchStream, F19,
        },
    };
    use ark_poly::multivariate::{SparsePolynomial, SparseTerm};

    #[test]
    fn to_evaluations_from_evaluations_sanity() {
        let p1: SparsePolynomial<F19, SparseTerm> = four_variable_polynomial::<F19>();
        let p1_evaluations: Vec<F19> = p1.to_evaluations();
        assert_eq!(
            p1,
            <SparsePolynomial<F19, SparseTerm> as Polynomial<F19>>::from_hypercube_evaluations(
                p1_evaluations
            )
        );

        let num_variables: usize = 16;
        let s: BenchStream<F19> = BenchStream::new(num_variables);
        let hypercube_len: usize = 2usize.pow(num_variables as u32);
        let mut p2_evaluations: Vec<F19> = Vec::with_capacity(hypercube_len);
        for i in 0..hypercube_len {
            p2_evaluations.push(s.evaluation(i));
        }
        let p2: SparsePolynomial<F19, SparseTerm> =
            <SparsePolynomial<F19, SparseTerm> as Polynomial<F19>>::from_hypercube_evaluations(
                p2_evaluations.clone(),
            );
        assert_eq!(p2_evaluations, p2.to_evaluations());
    }
}
