use ark_ff::define_field;
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

#[derive(MontConfig)]
#[modulus = "143244528689204659050391023439224324689"]
#[generator = "2"]
pub struct F128Config;
pub type F128 = Fp128<MontBackend<F128Config, 2>>;
