//! SIMD-vectorized field arithmetic using native intrinsics.
//!
//! Each base field provides platform-specific implementations of add, sub, mul
//! operating on packed SIMD vectors. Currently supports:
//!
//! - **Goldilocks** (p = 2^64 − 2^32 + 1) via NEON on aarch64, AVX-512 IFMA on x86_64.

pub mod goldilocks;

/// Platform-agnostic packed field operations.
///
/// Each ISA backend (NEON, AVX2, AVX-512) provides its own implementation
/// with the appropriate packed vector type.
///
/// # Safety
///
/// All values stored in `Packed` vectors must be valid field elements
/// (i.e., in `0..P`). The arithmetic functions maintain this invariant
/// when given valid inputs.
pub trait SimdBaseField: Copy + Send + Sync + Sized + 'static {
    /// Scalar representation (u32 for 31-bit fields, u64 for Goldilocks).
    type Scalar: Copy + Send + Sync + Default + PartialEq + core::fmt::Debug + 'static;

    /// The packed SIMD vector type (e.g., `uint64x2_t`, `__m256i`).
    type Packed: Copy;

    /// Number of scalar lanes in one `Packed` vector.
    const LANES: usize;

    /// The field modulus as a scalar.
    const MODULUS: Self::Scalar;

    /// Zero element.
    const ZERO: Self::Scalar;

    /// One element.
    const ONE: Self::Scalar;

    /// Broadcast a scalar to all lanes.
    fn splat(val: Self::Scalar) -> Self::Packed;

    /// Load a packed vector from a pointer (must be aligned to `Packed`).
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `LANES` valid `Scalar` values.
    unsafe fn load(ptr: *const Self::Scalar) -> Self::Packed;

    /// Store a packed vector to a pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to writable memory for at least `LANES` `Scalar` values.
    unsafe fn store(ptr: *mut Self::Scalar, v: Self::Packed);

    /// Packed modular addition: `(a + b) mod P`.
    fn add(a: Self::Packed, b: Self::Packed) -> Self::Packed;

    /// Packed modular subtraction: `(a - b) mod P`.
    fn sub(a: Self::Packed, b: Self::Packed) -> Self::Packed;

    /// Packed modular multiplication: `(a * b) mod P`.
    fn mul(a: Self::Packed, b: Self::Packed) -> Self::Packed;

    /// Scalar modular addition (non-vectorized, for reductions).
    fn scalar_add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Scalar modular subtraction (non-vectorized, for reductions).
    fn scalar_sub(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Scalar modular multiplication (non-vectorized, for reductions).
    fn scalar_mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Wrapping add without modular reduction — just raw integer addition.
    ///
    /// Callers must track carries separately and finalize with `reduce_carry`.
    /// Backends should override for performance; default falls back to `add`.
    #[inline(always)]
    fn add_wrapping(a: Self::Packed, b: Self::Packed) -> Self::Packed {
        Self::add(a, b)
    }

    /// Detect carries from a wrapping add: returns a packed vector with `1` in
    /// lanes where `sum < a` (unsigned overflow) and `0` elsewhere.
    ///
    /// Default returns zero (no carries tracked — consistent with `add` default).
    #[inline(always)]
    fn carry_mask(_sum: Self::Packed, _a_before: Self::Packed) -> Self::Packed {
        Self::splat(Self::ZERO)
    }

    /// Correct a wrapping accumulator given the carry count per lane.
    ///
    /// For Goldilocks: each carry represents 2^64 ≡ EPSILON (mod P),
    /// so result = sum + carry_count * EPSILON (mod P).
    ///
    /// Default is identity (assumes `add_wrapping` already reduced).
    #[inline(always)]
    fn reduce_carry(sum: Self::Packed, _carry_count: Self::Packed) -> Self::Packed {
        sum
    }

    /// Load `2 * LANES` scalars from interleaved pairs and deinterleave:
    ///   `[a0, b0, a1, b1, ..., a_{L-1}, b_{L-1}]` → `(evens, odds)`.
    ///
    /// Default: scalar deinterleave through stack buffers.
    /// Backends with native shuffle (e.g. AVX-512 `vpermutex2var`) should override.
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `2 * LANES` valid `Scalar` values.
    #[inline(always)]
    unsafe fn load_deinterleaved(ptr: *const Self::Scalar) -> (Self::Packed, Self::Packed) {
        let mut evens = [Self::ZERO; 16];
        let mut odds = [Self::ZERO; 16];
        for j in 0..Self::LANES {
            evens[j] = *ptr.add(2 * j);
            odds[j] = *ptr.add(2 * j + 1);
        }
        (Self::load(evens.as_ptr()), Self::load(odds.as_ptr()))
    }
}
