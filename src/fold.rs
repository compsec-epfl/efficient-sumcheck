//! The fold primitive (Thaler Lemma 4.3, equation 4.13).
//!
//! Half-split (MSB) fold: `new[k] = v[k] + weight · (v[k + L/2] − v[k])`.
//!
//! SIMD-accelerated for Goldilocks on NEON (aarch64) and AVX-512 IFMA
//! (x86_64). Falls back to scalar code for other fields. The detection
//! is compile-time constant-folded — zero overhead on the scalar path.
//!
//! This is exposed as a standalone public function because callers
//! (e.g., WHIR's `multilinear_fold`) need it independently of the full
//! sumcheck protocol.

// Re-export the existing implementation from multilinear_sumcheck.
// When Phase 6 lands and multilinear_sumcheck.rs is deleted, the fold
// implementation will move here directly.
pub use crate::multilinear_sumcheck::fold;
