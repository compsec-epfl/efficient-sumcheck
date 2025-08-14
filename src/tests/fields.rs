use ark_ff::fields::{Fp128, Fp64, MontBackend, MontConfig};

use crate::fields::small_fp_backend::SmallFp;
use crate::fields::small_fp_backend::SmallFpConfig;
use ark_ff::{BigInt, SqrtPrecomputation};
use fields_macro::SmallFpConfig;

#[derive(SmallFpConfig)]
#[modulus = "4294967295"]
#[generator = "2"]
pub struct SmallF32Config;
pub type SmallF32 = SmallFp<SmallF32Config>;

#[derive(SmallFpConfig)]
#[modulus = "18446744069414584321"] // q = 2^64 - 2^32 + 1
#[generator = "2"]
pub struct SmallF64Config;
pub type SmallF64 = SmallFp<SmallF64Config>;

#[derive(SmallFpConfig)]
#[modulus = "143244528689204659050391023439224324689"] // q = 143244528689204659050391023439224324689
#[generator = "2"]
pub struct SmallF128Config;
pub type SmallF128 = SmallFp<SmallF128Config>;

#[derive(MontConfig)]
#[modulus = "19"]
#[generator = "2"]
pub struct F19Config;
pub type F19 = Fp64<MontBackend<F19Config, 1>>;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"] // q = 2^64 - 2^32 + 1
#[generator = "2"]
pub struct F64Config;
pub type F64 = Fp64<MontBackend<F64Config, 1>>;

#[derive(MontConfig)]
#[modulus = "143244528689204659050391023439224324689"] // q = 143244528689204659050391023439224324689
#[generator = "2"]
pub struct F128Config;
pub type F128 = Fp128<MontBackend<F128Config, 2>>;
