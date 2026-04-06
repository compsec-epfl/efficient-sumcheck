//! Goldilocks field (p = 2^64 - 2^32 + 1) SIMD backends.

#[cfg(target_arch = "aarch64")]
pub mod mont_neon;

/// Montgomery-form Goldilocks backend (for both SmallFp and Fp64<MontBackend>).
/// Operates directly on arkworks' internal representation — zero-cost transmute.
#[cfg(target_arch = "aarch64")]
pub use mont_neon::MontGoldilocksNeon;
