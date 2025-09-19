use ark_ff::{One, PrimeField, Zero};

use crate::tests::fields::{SmallF19, SmallF19Mont, SmallF32, SmallF32Mont, F19};

#[test]
fn test_montgomery_basic_arithmetic() {
    // Test basic arithmetic operations remain consistent between standard and Montgomery
    let a_std = SmallF19::from(5u32);
    let b_std = SmallF19::from(7u32);

    let a_mont = SmallF19Mont::from(5u32);
    let b_mont = SmallF19Mont::from(7u32);

    // Addition
    let sum_std = a_std + b_std;
    let sum_mont = a_mont + b_mont;
    assert_eq!(
        sum_std.into_bigint(),
        sum_mont.into_bigint(),
        "Addition should be consistent"
    );

    // Multiplication
    let prod_std = a_std * b_std;
    let prod_mont = a_mont * b_mont;
    assert_eq!(
        prod_std.into_bigint(),
        prod_mont.into_bigint(),
        "Multiplication should be consistent"
    );

    // Subtraction
    let diff_std = a_std - b_std;
    let diff_mont = a_mont - b_mont;
    assert_eq!(
        diff_std.into_bigint(),
        diff_mont.into_bigint(),
        "Subtraction should be consistent"
    );
}

#[test]
fn test_montgomery_constants() {
    // Test that constants work correctly
    assert_eq!(
        SmallF19::zero().into_bigint(),
        SmallF19Mont::zero().into_bigint(),
        "Zero should be consistent"
    );
    assert_eq!(
        SmallF19::one().into_bigint(),
        SmallF19Mont::one().into_bigint(),
        "One should be consistent"
    );

    // Test that we get actual values when using into_bigint (not Montgomery form)
    assert_eq!(
        SmallF19Mont::zero().into_bigint().0[0],
        0,
        "Zero should display as 0"
    );
    assert_eq!(
        SmallF19Mont::one().into_bigint().0[0],
        1,
        "One should display as 1"
    );
}

#[test]
fn test_montgomery_display() {
    // Test that Display trait shows true mathematical values
    let val_std = SmallF19::from(13u32);
    let val_mont = SmallF19Mont::from(13u32);

    assert_eq!(
        val_std.to_string(),
        val_mont.to_string(),
        "Display should be consistent"
    );
    assert_eq!(
        val_mont.to_string(),
        "13",
        "Montgomery field should display true value"
    );
}

#[test]
fn test_montgomery_larger_fields() {
    // Test with larger fields
    let a_std = SmallF32::from(12345u32);
    let b_std = SmallF32::from(67890u32);

    let a_mont = SmallF32Mont::from(12345u32);
    let b_mont = SmallF32Mont::from(67890u32);

    let result_std = a_std * b_std + a_std;
    let result_mont = a_mont * b_mont + a_mont;

    assert_eq!(
        result_std.into_bigint(),
        result_mont.into_bigint(),
        "Complex operations should be consistent"
    );
    assert_eq!(
        result_std.to_string(),
        result_mont.to_string(),
        "Display should be consistent for larger fields"
    );
}

#[test]
fn test_montgomery_field_conversions() {
    // Test from_bigint and into_bigint round trip
    let original = SmallF19Mont::from(17u32);
    let bigint = original.into_bigint();
    let recovered = SmallF19Mont::from_bigint(bigint).unwrap();

    assert_eq!(
        original.into_bigint(),
        recovered.into_bigint(),
        "BigInt round trip should work"
    );
    assert_eq!(
        bigint.0[0] as u32, 17,
        "BigInt should contain true mathematical value"
    );
}

#[test]
fn test_montgomery_vs_arkworks() {
    // Test consistency with arkworks Montgomery implementation
    let val1 = 5u32;
    let val2 = 7u32;

    let our_mont = SmallF19Mont::from(val1) * SmallF19Mont::from(val2);
    let arkworks = F19::from(val1) * F19::from(val2);

    // Both should give the same mathematical result (convert arkworks to u64 for comparison)
    let our_result = our_mont.into_bigint().0[0];
    let arkworks_result = arkworks.into_bigint().0[0];
    assert_eq!(
        our_result, arkworks_result,
        "Should match arkworks Montgomery result"
    );
}
