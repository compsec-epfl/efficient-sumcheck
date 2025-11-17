use ark_ff::{
    ark_ff_macros::SmallFpConfig,
    fields::{Fp128, Fp64, MontBackend, MontConfig},
    BigInt, SmallFp, SmallFpConfig,
};
use ark_ff::{Fp2, Fp2Config, Fp4, Fp4Config, SqrtPrecomputation};
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
#[modulus = "65521"]
#[generator = "2"]
#[backend = "montgomery"]
pub struct SmallF16ConfigMont;
pub type SmallF16 = SmallFp<SmallF16ConfigMont>;

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

// SmallM31 extensions
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp2SmallM31Config;

impl Fp2Config for Fp2SmallM31Config {
    type Fp = SmallM31;

    // Use const_new to build compile-time constants
    const NONRESIDUE: SmallM31 = SmallM31::new(3);

    // These Frobenius coeffs aren't used for arithmetic benchmarks anyway
    const FROBENIUS_COEFF_FP2_C1: &'static [SmallM31] = &[SmallM31::new(1), SmallM31::new(3)];
}

pub type Fp2SmallM31 = Fp2<Fp2SmallM31Config>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp4SmallM31Config;

impl Fp4Config for Fp4SmallM31Config {
    type Fp2Config = Fp2SmallM31Config;

    const NONRESIDUE: Fp2<Fp2SmallM31Config> =
        Fp2::<Fp2SmallM31Config>::new(SmallM31::new(3), SmallM31::new(0));

    // 👇 now a slice of base‐field elements, not Fp2 elements
    const FROBENIUS_COEFF_FP4_C1: &'static [SmallM31] = &[
        SmallM31::new(1),
        SmallM31::new(3),
        SmallM31::new(9),
        SmallM31::new(27),
    ];
}

pub type Fp4SmallM31 = Fp4<Fp4SmallM31Config>;
