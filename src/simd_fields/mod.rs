//! SIMD-vectorized field arithmetic using native intrinsics.
//!
//! Each base field provides platform-specific implementations of add, sub, mul
//! operating on packed SIMD vectors.

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
}

/// Bridge trait: connects an arkworks `Field` type to its SIMD backend.
///
/// Implement this for any arkworks field type (e.g., `Fp64<MontBackend<F64Config, 1>>`)
/// to enable compile-time dispatch to the SIMD sumcheck path.
///
/// The conversion functions handle the representation difference
/// (e.g., Montgomery form → canonical) at the sumcheck boundary.
/// This is an O(n) one-time cost that's amortized over the O(n log n) sumcheck.
pub trait SimdAccelerated: ark_ff::Field + Sized {
    /// The SIMD backend for this field.
    type Backend: SimdBaseField;

    /// Convert from arkworks field element to raw scalar.
    fn to_raw(val: Self) -> <Self::Backend as SimdBaseField>::Scalar;

    /// Convert from raw scalar to arkworks field element.
    fn from_raw(val: <Self::Backend as SimdBaseField>::Scalar) -> Self;

    /// Bulk convert a slice of arkworks elements to raw scalars.
    ///
    /// Default implementation calls `to_raw` element-wise.
    /// Override for zero-cost `transmute` when the representations match
    /// (e.g., `SmallFp` backends where internal repr IS the canonical value).
    fn slice_to_raw(src: &[Self]) -> Vec<<Self::Backend as SimdBaseField>::Scalar> {
        src.iter().map(|x| Self::to_raw(*x)).collect()
    }

    /// Bulk convert raw scalars back to arkworks elements.
    fn slice_from_raw(src: &[<Self::Backend as SimdBaseField>::Scalar]) -> Vec<Self> {
        src.iter().map(|x| Self::from_raw(*x)).collect()
    }
}
