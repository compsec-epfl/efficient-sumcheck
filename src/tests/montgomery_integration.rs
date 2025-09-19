use crate::tests::{
    fields::{SmallF19Mont, SmallF32Mont},
    polynomials::three_variable_polynomial_evaluations,
};
use ark_ff::Zero;

#[test]
fn test_montgomery_polynomial_evaluation() {
    // Test that polynomial evaluations work correctly with Montgomery
    let evaluations = three_variable_polynomial_evaluations::<SmallF19Mont>();

    // Verify we get the expected number of evaluations
    assert_eq!(evaluations.len(), 8, "Should have 2^3 = 8 evaluations");

    // Verify that the evaluations are not all zero (would indicate a problem)
    let non_zero_count = evaluations.iter().filter(|&&x| !x.is_zero()).count();
    assert!(non_zero_count > 0, "Should have some non-zero evaluations");

    // Test that we can sum the evaluations (exercises field arithmetic)
    let sum = evaluations
        .iter()
        .fold(SmallF19Mont::zero(), |acc, &x| acc + x);

    // The sum should be 6 for this particular polynomial over F_19
    // 4*x_1*x_2 + 7*x_2*x_3 + 2*x_1 + 13*x_2 evaluated at all boolean points sums to 6
    assert_eq!(sum, SmallF19Mont::from(6u32), "Sum should be 6");
}
