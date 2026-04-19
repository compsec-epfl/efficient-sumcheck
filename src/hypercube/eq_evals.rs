use ark_ff::Field;

/// Compute eq(τ, ·) over {0,1}^v using the incremental build-up algorithm.
///
/// `eq(x, y) = Π_j (x_j · y_j + (1 - x_j)(1 - y_j))`.
///
/// Returns a table of length `2^v` where entry `i` is `eq(τ, y_i)` with
/// `y_i` the Boolean vector whose bit `j` is `(i >> j) & 1` (LSB-indexed).
///
/// Complexity: O(2^v) — one multiply per entry. The algorithm doubles the
/// table one variable at a time: for each `τ_j`, existing entries are
/// split into `entry * (1 - τ_j)` (bit 0) and `entry * τ_j` (bit 1).
pub fn compute_hypercube_eq_evals<F: Field>(num_variables: usize, point: &[F]) -> Vec<F> {
    let size = 1 << num_variables;
    let mut table = Vec::with_capacity(size);
    table.push(F::one());

    for &tau_j in point[..num_variables].iter().rev() {
        let len = table.len();
        let one_minus = F::one() - tau_j;
        // Process in reverse so we can expand in place.
        table.resize(2 * len, F::zero());
        for i in (0..len).rev() {
            table[2 * i + 1] = table[i] * tau_j;
            table[2 * i] = table[i] * one_minus;
        }
    }

    table
}

/// Evaluate `eq(τ, y)` at a single Boolean point `y ∈ {0,1}^v`.
///
/// `point` is the integer whose bit `j` is `y_j` (LSB-indexed).
/// This is O(v) — use it when you need one entry instead of the full
/// table from [`compute_hypercube_eq_evals`].
///
/// `eq_poly(τ, i) == compute_hypercube_eq_evals(τ.len(), τ)[i]`.
pub fn eq_poly<F: Field>(tau: &[F], point: usize) -> F {
    let num_variables = tau.len();
    (0..num_variables).fold(F::one(), |acc, j| {
        if (point >> j) & 1 == 1 {
            acc * tau[j]
        } else {
            acc * (F::one() - tau[j])
        }
    })
}

/// Evaluate `eq(x, y)` where both `x` and `y` are field element vectors.
///
/// `eq(x, y) = Π_j (x_j · y_j + (1 − x_j)(1 − y_j))`.
///
/// Unlike [`eq_poly`] which takes a Boolean point as an integer, this
/// handles non-binary evaluation points — needed for oracle checks in
/// composed protocols (WARP, GKR reduce-to-one).
pub fn eq_poly_non_binary<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).fold(F::one(), |acc, (x_i, y_i)| {
        acc * (*x_i * *y_i + (F::one() - x_i) * (F::one() - y_i))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::F64;

    #[test]
    fn test_eq_evals_2_variables() {
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

    #[test]
    fn test_eq_poly_matches_table() {
        let tau = vec![F64::from(2u64), F64::from(3u64)];
        let table = compute_hypercube_eq_evals(2, &tau);
        for i in 0..4 {
            assert_eq!(eq_poly(&tau, i), table[i], "mismatch at point {i}");
        }
    }

    #[test]
    fn test_eq_poly_non_binary_matches_table_on_binary() {
        // When y is binary, eq_poly_non_binary should match eq_poly.
        let tau = vec![F64::from(2u64), F64::from(3u64)];
        for i in 0..4usize {
            let y: Vec<F64> = (0..2).map(|j| F64::from(((i >> j) & 1) as u64)).collect();
            assert_eq!(
                eq_poly_non_binary(&tau, &y),
                eq_poly(&tau, i),
                "mismatch at point {i}"
            );
        }
    }

    #[test]
    fn test_eq_poly_non_binary_symmetric() {
        // eq(x, y) == eq(y, x)
        let x = vec![F64::from(5u64), F64::from(7u64), F64::from(3u64)];
        let y = vec![F64::from(11u64), F64::from(2u64), F64::from(9u64)];
        assert_eq!(eq_poly_non_binary(&x, &y), eq_poly_non_binary(&y, &x));
    }
}
