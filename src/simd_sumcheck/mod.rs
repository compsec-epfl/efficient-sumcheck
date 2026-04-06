//! SIMD-vectorized sumcheck algorithm layer.
//!
//! Generic over [`SimdBaseField`](super::simd_fields::SimdBaseField).

pub(crate) mod dispatch;
pub mod evaluate;
pub mod prove;
pub mod reduce;
