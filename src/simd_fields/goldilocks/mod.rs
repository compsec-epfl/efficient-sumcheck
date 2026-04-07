//! Goldilocks field (p = 2^64 - 2^32 + 1) SIMD backends.

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
pub mod avx512;

/// Goldilocks NEON backend (aarch64).
///
/// Operates on Montgomery-form values as stored by arkworks (`SmallFp.value`
/// or `Fp64.0.0[0]`) — zero-cost transmute from `&[Field]` to `&[u64]`.
#[cfg(target_arch = "aarch64")]
pub use neon::GoldilocksNeon;

/// Goldilocks AVX-512 IFMA backend (x86_64).
///
/// Same Montgomery-form transmute as the NEON backend, but with true 8-wide
/// vectorized multiplication via 52-bit IFMA decomposition.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
pub use avx512::GoldilocksAvx512;
