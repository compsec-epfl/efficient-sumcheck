use ark_ff::Field;

use crate::hypercube::Hypercube;
use crate::order_strategy::AscendingOrder;

/// Compute eq(point, ·) over {0,1}^num_variables.
///
/// `eq(x, y) = Π_j (x_j · y_j + (1 - x_j)(1 - y_j))`.
pub fn compute_hypercube_eq_evals<F: Field>(num_variables: usize, point: &[F]) -> Vec<F> {
    Hypercube::<AscendingOrder>::new(num_variables)
        .map(|(index, _)| {
            (0..num_variables).fold(F::one(), |acc, j| {
                let bit = F::from((index >> j & 1) as u64);
                acc * (point[j] * bit + (F::one() - point[j]) * (F::one() - bit))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::F64;

    #[test]
    fn test_eq_evals_2_variables() {
        // point = [2, 3], ascending bit order
        // index 0 = (0,0): (-1)·(-2) = 2
        // index 1 = (1,0): (2)·(-2) = -4
        // index 2 = (0,1): (-1)·(3) = -3
        // index 3 = (1,1): (2)·(3) = 6
        let point = vec![F64::from(2u64), F64::from(3u64)];
        let evals = compute_hypercube_eq_evals(2, &point);

        assert_eq!(evals.len(), 4);
        assert_eq!(evals[0], F64::from(2u64));
        assert_eq!(evals[1], -F64::from(4u64));
        assert_eq!(evals[2], -F64::from(3u64));
        assert_eq!(evals[3], F64::from(6u64));
    }

    #[test]
    fn test_eq_evals_sum_is_one_on_binary_point() {
        // eq(b, ·) is 1 at b and 0 elsewhere
        let point = vec![F64::from(1u64), F64::from(0u64), F64::from(1u64)];
        let evals = compute_hypercube_eq_evals(3, &point);

        assert_eq!(evals.len(), 8);

        let point_index = 0b101;
        assert_eq!(evals[point_index], F64::from(1u64));
        for (i, &e) in evals.iter().enumerate() {
            if i != point_index {
                assert_eq!(e, F64::from(0u64));
            }
        }
    }
}
