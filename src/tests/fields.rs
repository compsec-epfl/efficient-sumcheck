use ark_ff::define_field;
use ark_ff::fields::models::cubic_extension::{CubicExtConfig, CubicExtField};
use ark_ff::fields::models::quadratic_extension::{QuadExtConfig, QuadExtField};
use ark_ff::fields::{Fp128, Fp64, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "19"]
#[generator = "2"]
pub struct F19Config;
pub type F19 = Fp64<MontBackend<F19Config, 1>>;

#[derive(MontConfig)]
#[modulus = "2147483647"] // 2 ^ 31 - 1
#[generator = "2"]
pub struct M31Config;
pub type M31 = Fp64<MontBackend<M31Config, 1>>;

#[derive(MontConfig)]
#[modulus = "2013265921"] // BabyBear 2^{31} - 2^{27} + 1
#[generator = "2"]
pub struct BabyBearConfig;
pub type BabyBear = Fp64<MontBackend<BabyBearConfig, 1>>;

// Goldilocks: q = 2^64 - 2^32 + 1
// Primary type: SmallFp (optimal single-u64 Montgomery representation).
define_field!(
    modulus = "18446744069414584321",
    generator = "7",
    name = F64,
);

/// Extract the raw Montgomery-form `u64` from a Goldilocks field element.
pub fn to_mont(f: F64) -> u64 {
    f.value
}

/// Reconstruct an `F64` from its raw Montgomery-form `u64`.
pub fn from_mont(val: u64) -> F64 {
    F64::from_raw(val)
}

// Secondary type: Fp64<MontBackend> (for compatibility with code using MontConfig).
// Both F64 and FpF64 store a single u64 in Montgomery form — the SIMD backend
// works identically for either.
#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct FpF64Config;
pub type FpF64 = Fp64<MontBackend<FpF64Config, 1>>;

// Degree-2 extension of Goldilocks: F64[X] / (X² - 7)
// NONRESIDUE = 7 (must be a non-square in F64).
pub struct F64Ext2Config;
impl QuadExtConfig for F64Ext2Config {
    type BasePrimeField = F64;
    type BaseField = F64;
    type FrobCoeff = F64;
    const DEGREE_OVER_BASE_PRIME_FIELD: usize = 2;
    const NONRESIDUE: F64 = F64::from_raw(7);
    // Frobenius coefficient: NONRESIDUE^((p-1)/2). For testing, -1 works
    // for any non-square nonresidue (Euler criterion).
    // Frobenius coefficients: [1, -1].
    // -1 mod P = P - 1 = 0xFFFF_FFFF_0000_0000 in Montgomery form.
    // Actually, -1 in Montgomery form is mont(-1) = mont(P-1) = (P-1)*R mod P.
    // For Goldilocks, R mod P = EPSILON = 0xFFFFFFFF.
    // mont(P-1) = (P-1) * R mod P. Let's just use from_raw(P - 1)... no.
    // from_raw takes a value already in Montgomery form.
    // -1 in Montgomery form = R * (P-1) mod P = (-R) mod P = P - EPSILON = P - (2^32-1)
    //   = 0xFFFF_FFFF_0000_0001 - 0xFFFF_FFFF = 0xFFFF_FFFE_0000_0002
    // Actually easier: just use the constant P - EPSILON.
    const FROBENIUS_COEFF_C1: &'static [F64] = &[
        F64::from_raw(0xFFFF_FFFF),           // mont(1) = R mod P = EPSILON
        F64::from_raw(0xFFFF_FFFE_0000_0002), // mont(-1) = P - EPSILON
    ];

    fn mul_base_field_by_frob_coeff(fe: &mut Self::BaseField, power: usize) {
        *fe *= &Self::FROBENIUS_COEFF_C1[power % 2];
    }
}
pub type F64Ext2 = QuadExtField<F64Ext2Config>;

// Degree-3 extension of Goldilocks: F64[X] / (X³ - 7)
pub struct F64Ext3Config;
impl CubicExtConfig for F64Ext3Config {
    type BasePrimeField = F64;
    type BaseField = F64;
    type FrobCoeff = F64;
    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<CubicExtField<Self>>> = None;
    const DEGREE_OVER_BASE_PRIME_FIELD: usize = 3;
    const NONRESIDUE: F64 = F64::from_raw(7);
    // Frobenius coefficients for cubic extension.
    // FROBENIUS_COEFF_C1[i] = NONRESIDUE^((p^i - 1) / 3)
    // FROBENIUS_COEFF_C2[i] = NONRESIDUE^((2*(p^i - 1)) / 3)
    // For testing purposes, we use identity (power 0) and compute the rest.
    // Since p ≡ 1 mod 3 for Goldilocks, these exist.
    // For simplicity, use [1, w^((p-1)/3), w^(2(p-1)/3)] but computing these
    // requires modular exponentiation. For test-only usage, just provide placeholders
    // that satisfy the trait — the sumcheck doesn't use Frobenius.
    const FROBENIUS_COEFF_C1: &'static [F64] = &[F64::from_raw(0xFFFF_FFFF)]; // [1]
    const FROBENIUS_COEFF_C2: &'static [F64] = &[F64::from_raw(0xFFFF_FFFF)]; // [1]

    fn mul_base_field_by_frob_coeff(
        _c1: &mut Self::BaseField,
        _c2: &mut Self::BaseField,
        _power: usize,
    ) {
        // Frobenius not used in sumcheck — no-op for testing
    }
}
pub type F64Ext3 = CubicExtField<F64Ext3Config>;

#[derive(MontConfig)]
#[modulus = "143244528689204659050391023439224324689"]
#[generator = "2"]
pub struct F128Config;
pub type F128 = Fp128<MontBackend<F128Config, 2>>;
