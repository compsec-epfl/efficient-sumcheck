use ark_ff::{
    ark_ff_macros::SmallFpConfig,
    fields::{Fp128, Fp64, MontBackend, MontConfig},
    BigInt, SmallFp,
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
        Fp2::<Fp2SmallM31Config>::new(SmallM31::new(3), SmallM31::new(7));

    // 👇 now a slice of base‐field elements, not Fp2 elements
    const FROBENIUS_COEFF_FP4_C1: &'static [SmallM31] = &[
        SmallM31::new(1),
        SmallM31::new(3),
        SmallM31::new(9),
        SmallM31::new(27),
    ];
}

pub type Fp4SmallM31 = Fp4<Fp4SmallM31Config>;

// SmallGoldilocks extensions
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp2SmallGoldilocksConfig;

impl Fp2Config for Fp2SmallGoldilocksConfig {
    type Fp = SmallGoldilocks;

    /// For Goldilocks, 7 is a quadratic non-residue.
    /// The extension is formed by SmallGoldilocks[u] / (u^2 - 7)
    /// Reference: https://github.com/zhenfeizhang/Goldilocks
    const NONRESIDUE: SmallGoldilocks = SmallGoldilocks::new(7);

    /// Frobenius coefficients are used for computing elements raised to the power of the modulus.
    /// In a quadratic extension, these are often just [1, -1] or precomputed constants.
    const FROBENIUS_COEFF_FP2_C1: &'static [SmallGoldilocks] = &[
        SmallGoldilocks::new(1),
        // This is typically -1 in the field, or the precomputed Frobenius constant.
        SmallGoldilocks::new(18446744069414584320),
    ];
}

pub type Fp2SmallGoldilocks = Fp2<Fp2SmallGoldilocksConfig>;



#[cfg(test)]
mod tests {
    use super::*;

    fn as_bytes<T>(v: &T) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                (v as *const T) as *const u8,
                core::mem::size_of::<T>(),
            )
            .to_vec()
        }
    }

    #[test]
    fn smallgoldilocks_vs_goldilocks_raw_and_fmt_should_be_the_same() {
        // SmallFp has `new`, Fp64 commonly supports From<u64>.
        let small = SmallGoldilocks::new(7);
        let normal = F64::from(7u64);

        // Display formatting: should be canonical "7" for both.
        let small_fmt = format!("{}", small); // <-- bug that this IS mont form
        let normal_fmt = format!("{}", normal); // <-- this is 7 (as expected)
        assert_eq!(small_fmt, normal_fmt);

        // whatever is in memory is the same 
        let small_bytes = as_bytes(&small); // <-- TODO: bug that this is NOT mont form? (it's 7)
        let normal_bytes = as_bytes(&normal); // <-- this is mont form (as expected)
        assert_eq!(
            small_bytes, normal_bytes,
            "Raw in-memory bytes differ between SmallGoldilocks and Goldilocks for 7."
        );
    }

    #[test]
    fn debug_smallf16_and_smallm31_and_goldilocks() {

        // --- SmallF16 ---
        println!("========================================");
        println!("Testing SmallF16 (Modulus 65521)");
        println!("========================================");
        
        let small_f16 = SmallF16::new(7);
        let f16_fmt = format!("{}", small_f16);
        let f16_bytes = as_bytes(&small_f16);

        println!("Display Formatted: {}", f16_fmt);
        println!("Raw Bytes (Hex):   {:02x?}", f16_bytes);
        

        // --- SmallM31 ---
        println!("\n========================================");
        println!("Testing SmallM31 (Modulus 2^31 - 1)");
        println!("========================================");

        let small_m31 = SmallM31::new(7);
        let m31_fmt = format!("{}", small_m31);
        let m31_bytes = as_bytes(&small_m31);

        println!("Display Formatted: {}", m31_fmt);
        println!("Raw Bytes (Hex):   {:02x?}", m31_bytes);


        // --- SmallGoldilocks ---
        println!("\n========================================");
        println!("Testing SmallGoldilocks (Modulus 2^64 - 2^32 + 1)");
        println!("========================================");

        let small_gld = SmallGoldilocks::new(7);
        let gld_fmt = format!("{}", small_gld);
        let gld_bytes = as_bytes(&small_gld);

        println!("Display Formatted: {}", gld_fmt);
        println!("Raw Bytes (Hex):   {:02x?}", gld_bytes);
        
    }
}