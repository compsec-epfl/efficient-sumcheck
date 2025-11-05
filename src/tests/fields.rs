use ark_ff::{BigInt, SmallFp, SmallFpConfig, ark_ff_macros::SmallFpConfig, fields::{Fp64, Fp128, MontBackend, MontConfig}};
use ark_ff::SqrtPrecomputation;
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
#[modulus = "18446744069414584321"] // q = 2^64 - 2^32 + 1
#[generator = "2"]
pub struct F64Config;
pub type F64 = Fp64<MontBackend<F64Config, 1>>;

#[derive(MontConfig)]
#[modulus = "143244528689204659050391023439224324689"] // q = 143244528689204659050391023439224324689
#[generator = "2"]
pub struct F128Config;
pub type F128 = Fp128<MontBackend<F128Config, 2>>;

#[derive(SmallFpConfig)]
#[modulus = "2147483647"] // 2 ^ 31 - 1
#[generator = "2"]
#[backend = "montgomery"]
pub struct SmallM31ConfigMont;
pub type SmallM31 = SmallFp<SmallM31ConfigMont>;

#[derive(SmallFpConfig)]
#[modulus = "18446744069414584321"] // Goldilock's prime 2^64 - 2^32 + 1
#[generator = "2"]
#[backend = "montgomery"]
pub struct SmallF64ConfigMont;
pub type SmallGoldilocks = SmallFp<SmallF64ConfigMont>;