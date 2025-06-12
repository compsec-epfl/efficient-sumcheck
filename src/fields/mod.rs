mod const_helpers;
pub mod fp_backend;
mod m31;
pub mod small_fp_backend;
mod vec_ops;

#[cfg(target_arch = "aarch64")]
pub mod aarch64_neon;

pub use m31::{reduce_sum_naive, M31, M31_MODULUS};
pub use vec_ops::VecOps;
