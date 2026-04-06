//! Goldilocks field (p = 2^64 - 2^32 + 1) SIMD backends.

#[cfg(target_arch = "aarch64")]
pub mod neon;

/// Goldilocks NEON backend (aarch64).
///
/// Operates on Montgomery-form values as stored by arkworks (`SmallFp.value`
/// or `Fp64.0.0[0]`) — zero-cost transmute from `&[Field]` to `&[u64]`.
#[cfg(target_arch = "aarch64")]
pub use neon::GoldilocksNeon;
