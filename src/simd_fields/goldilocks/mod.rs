//! Goldilocks field (p = 2^64 - 2^32 + 1) SIMD backends.

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "aarch64")]
pub mod mont_neon;

pub mod bridge;

/// Canonical-form Goldilocks backend (for SmallFp or direct representation).
#[cfg(target_arch = "aarch64")]
pub use neon::GoldilocksNeon as GoldilocksSIMD;

/// Montgomery-form Goldilocks backend (for Fp64<MontBackend<...>>).
/// Enables zero-cost `transmute` from arkworks field elements.
#[cfg(target_arch = "aarch64")]
pub use mont_neon::MontGoldilocksNeon as MontGoldilocksSIMD;
