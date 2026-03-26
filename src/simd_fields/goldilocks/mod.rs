//! Goldilocks field (p = 2^64 - 2^32 + 1) SIMD backends.

#[cfg(target_arch = "aarch64")]
pub mod neon;

pub mod bridge;

#[cfg(target_arch = "aarch64")]
pub use neon::GoldilocksNeon as GoldilocksSIMD;
