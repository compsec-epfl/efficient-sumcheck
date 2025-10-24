use crate::{
    hypercube::{Hypercube, HypercubeMember},
    order_strategy::GraycodeOrder,
};
use ark_ff::Field;
use ark_poly::{
    multivariate::{self, SparsePolynomial, SparseTerm, Term},
    DenseMVPolynomial,
};

/*
 * These are two small polynomials to use for sanity checking.
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
    // 4*x_1*x_2 + 7*x_2*x_3 + 2*x_1 + 13*x_2
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
    // 4*x_1*x_2 + 7*x_2*x_3 + 2*x_1 + 13*x_2 + 1x_4
    four_variable_polynomial().to_evaluations()
}

/*
 * Below here, we "extend" multivariate::SparsePolynomial<F, SparseTerm> so that we can
 * get evaluations over the boolean hypercube (it's not so important it's just handy for testing)
 *
 * The idea comes from here: https://github.com/montekki/thaler-study/blob/master/sum-check-protocol/src/lib.rs
 */

pub trait Polynomial<F: Field> {
    // Evaluates the polynomial at the provided point (expressed as a vector of field elements)
    fn evaluate(&self, point: Vec<F>) -> Option<F>;

    // Evaluates the polynomial at the provided point (expressed as a hypercube member)
    // using the given number of variables.
    fn evaluate_from_hypercube(&self, num_vars: usize, point: HypercubeMember) -> Option<F>;

    // Converts the polynomial into a vector containing evaluations at every
    // point of the hypercube.
    fn to_evaluations(&self) -> Vec<F>;

    // take the evaluations table and give back a sparsepolynomial
    fn from_hypercube_evaluations(evaluations: Vec<F>) -> SparsePolynomial<F, SparseTerm>;
}

impl<F: Field> Polynomial<F> for SparsePolynomial<F, SparseTerm> {
    fn evaluate(&self, point: Vec<F>) -> Option<F> {
        assert_eq!(DenseMVPolynomial::<F>::num_vars(self), point.len());
        // Compute the evaluation by summing the contributions of each term.
        let mut result = F::ZERO;
        for (coefficient, term) in self.terms().iter() {
            result += term.evaluate(&point) * coefficient;
        }

        Some(result)
    }

    fn evaluate_from_hypercube(&self, num_vars: usize, point: HypercubeMember) -> Option<F> {
        // Convert the boolean representation into field elements.
        let mut field_values: Vec<F> = Vec::with_capacity(num_vars);
        for bit in point {
            field_values.push(if bit { F::ONE } else { F::ZERO });
        }

        // Compute the evaluation by summing the contributions of each term.
        let mut result = F::ZERO;
        for (coefficient, term) in self.terms().iter() {
            result += term.evaluate(&field_values) * coefficient;
        }

        Some(result)
    }

    fn to_evaluations(&self) -> Vec<F> {
        let num_vars = DenseMVPolynomial::<F>::num_vars(self);
        let total_points = Hypercube::<GraycodeOrder>::stop_value(num_vars);
        let mut evaluations = Vec::with_capacity(total_points);

        // Iterate through each index of the hypercube.
        for index in 0..total_points {
            let point = HypercubeMember::new(num_vars, index);
            let value = Self::evaluate_from_hypercube(self, num_vars, point).unwrap();
            evaluations.push(value);
        }

        evaluations
    }

    // TODO (z-tech): this works but it's super slow
    fn from_hypercube_evaluations(mut evaluations: Vec<F>) -> SparsePolynomial<F, SparseTerm> {
        // Ensure that the evaluations vector length is a power of two.
        assert!(
            evaluations.len().is_power_of_two(),
            "evaluations len must be a power of two"
        );
        let num_vars: usize = evaluations.len().ilog2() as usize;
        let n = evaluations.len();

        // In-place bit reversal permutation:
        // If the evaluations were produced with the highest-index variable corresponding to the LSB,
        // we need to swap elements so that the i-th bit corresponds to variable x_i.
        for i in 0_usize..n {
            // Reverse the lower `num_vars` bits of i.
            let j = i.reverse_bits() >> (usize::BITS - num_vars as u32);
            if i < j {
                evaluations.swap(i, j);
            }
        }

        // Perform in-place Möbius inversion on `evaluations` (now in standard binary order).
        for i in 0..num_vars {
            for mask in 0..n {
                if mask & (1 << i) != 0 {
                    evaluations[mask] = evaluations[mask] - evaluations[mask ^ (1 << i)];
                }
            }
        }

        // Build the sparse polynomial representation from the nonzero coefficients.
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
        // we should get back the same polynomial
        let p1: SparsePolynomial<F19, SparseTerm> = four_variable_polynomial::<F19>();
        let p1_evaluations: Vec<F19> = p1.to_evaluations();
        assert_eq!(
            p1,
            <SparsePolynomial<F19, SparseTerm> as Polynomial<F19>>::from_hypercube_evaluations(
                p1_evaluations
            )
        );

        // we should get back the same evaluations
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
