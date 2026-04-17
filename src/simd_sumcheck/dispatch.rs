//! SIMD auto-dispatch for the multilinear sumcheck protocol.
//!
//! When `BF == EF` and both are a Goldilocks field (p = 2^64 − 2^32 + 1)
//! stored as a single `u64` in Montgomery form, the sumcheck is transparently
//! routed to a SIMD-accelerated backend:
//!
//! - **aarch64**: NEON backend (2-wide, scalar mul fallback)
//! - **x86_64 + AVX-512 IFMA**: AVX-512 backend (8-wide, true IFMA mul)
//!
//! Detection uses [`Field::BasePrimeField::MODULUS`] from arkworks — no
//! concrete type names are referenced. After monomorphization the check
//! is constant-folded by LLVM, so the dead branch is eliminated entirely.
//!
//! # Safety: `transmute_copy` between `Field` and `u64`
//!
//! The `u64_to_field` and `field_to_u64` helpers use `transmute_copy` to
//! reinterpret between arkworks field elements and raw Montgomery-form `u64`
//! values. This is safe for Goldilocks because:
//!
//! 1. `is_goldilocks()` verifies: extension degree == 1, `size_of::<F>()` == 8,
//!    modulus bits == 64, and modulus value == `0xFFFF_FFFF_0000_0001`.
//! 2. Both `SmallFp<P>` and `Fp64<MontBackend<_, 1>>` store a single `u64`
//!    as their only non-ZST field (`value: u64` resp. `BigInt<1>([u64; 1])`).
//!
//! This invariant is NOT guaranteed by `#[repr(transparent)]` in arkworks.
//! If arkworks changes the internal layout of these types, the SIMD path
//! must be updated. The `size_of` check provides a compile-time safety net.

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
use ark_ff::Field;

/// Goldilocks modulus: p = 2^64 − 2^32 + 1.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Returns `true` when `F` is a Goldilocks prime field stored as a
/// single `u64` in Montgomery form.
///
/// The check uses only the [`Field`] trait (via `BasePrimeField: PrimeField`):
///
/// 1. `extension_degree() == 1` — must be a prime field, not an extension.
/// 2. `size_of::<F>() == 8` — the element must be a single `u64`
///    (true for both `SmallFp<P>` and `Fp64<MontBackend<_, 1>>`).
/// 3. The modulus value equals `GOLDILOCKS_P`.
///
/// After monomorphization every operand is a compile-time constant,
/// so LLVM folds the entire function to `true` or `false`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
fn is_goldilocks<F: Field>() -> bool {
    use ark_ff::PrimeField;

    if F::extension_degree() != 1 {
        return false;
    }
    if core::mem::size_of::<F>() != core::mem::size_of::<u64>() {
        return false;
    }
    if F::BasePrimeField::MODULUS_BIT_SIZE != 64 {
        return false;
    }
    let modulus = F::BasePrimeField::MODULUS;
    let limbs: &[u64] = modulus.as_ref();
    limbs[0] == GOLDILOCKS_P && limbs[1..].iter().all(|&x| x == 0)
}

/// Returns `true` when `F` has Goldilocks as its base prime field,
/// regardless of extension degree. For degree-1 this is the same as
/// `is_goldilocks`. For degree 2, 3, etc., the element is `d` consecutive
/// `u64` values in Montgomery form.
///
/// After monomorphization, fully constant-folded by LLVM.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
fn is_goldilocks_based<F: Field>() -> bool {
    use ark_ff::PrimeField;

    if F::BasePrimeField::MODULUS_BIT_SIZE != 64 {
        return false;
    }
    // Check element size matches d * 8 bytes (d u64 components)
    let d = F::extension_degree() as usize;
    if core::mem::size_of::<F>() != d * core::mem::size_of::<u64>() {
        return false;
    }
    let modulus = F::BasePrimeField::MODULUS;
    let limbs: &[u64] = modulus.as_ref();
    limbs[0] == GOLDILOCKS_P && limbs[1..].iter().all(|&x| x == 0)
}

/// Extract the degree-2 nonresidue `w` from the extension field config.
/// Computes `(0, 1) * (0, 1) = (w, 0)` so `w` is at component 0.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline]
pub(crate) fn extract_nonresidue_ext2<
    EF: Field,
    S: crate::simd_fields::SimdBaseField<Scalar = u64>,
>() -> u64 {
    let one_x = unsafe {
        let mut tmp = [0u64; 2];
        tmp[1] = S::ONE;
        let one_x: EF = core::mem::transmute_copy(&tmp);
        one_x
    };
    let nr = one_x * one_x;
    unsafe { *((&nr) as *const EF as *const u64) }
}

/// Extract the degree-3 nonresidue `w` from the extension field config.
/// Computes `(0, 1, 0)^3 = X^3 = w` so `w` is at component 0.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline]
pub(crate) fn extract_nonresidue_ext3<
    EF: Field,
    S: crate::simd_fields::SimdBaseField<Scalar = u64>,
>() -> u64 {
    let one_x = unsafe {
        let mut tmp = [0u64; 3];
        tmp[1] = S::ONE;
        let one_x: EF = core::mem::transmute_copy(&tmp);
        one_x
    };
    let nr = one_x * one_x * one_x;
    unsafe { *((&nr) as *const EF as *const u64) }
}

// ─── Standalone SIMD reduce (Field-level API) ──────────────────────────────

/// SIMD-accelerated pairwise reduce on a `Vec<F>`.
///
/// If `F` is a recognised Goldilocks field, runs the SIMD reduce in-place
/// and truncates the vector. Otherwise returns `false` and the caller
/// should fall back to the generic path.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_reduce<F: Field>(evals: &mut Vec<F>, challenge: F) -> bool {
    if !is_goldilocks::<F>() {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_sumcheck::reduce::reduce_in_place;

    let buf: &mut [u64] =
        unsafe { core::slice::from_raw_parts_mut(evals.as_mut_ptr() as *mut u64, evals.len()) };
    let chg: u64 = field_to_u64(challenge);
    let new_len = reduce_in_place::<Backend>(buf, chg);
    evals.truncate(new_len);
    true
}

// ─── SIMD degree-1 evaluate for coefficient sumcheck ────────────────────────

/// Fused SIMD reduce + degree-1 evaluate.
///
/// Reduces `pw` in-place and returns `[s0, s1 - s0]` for the next round,
/// computed in a single data pass via `reduce_and_evaluate`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_fused_reduce_evaluate_degree1<F: Field>(
    pw: &mut Vec<F>,
    challenge: F,
) -> Option<Vec<F>> {
    if !is_goldilocks::<F>() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_sumcheck::reduce::reduce_and_evaluate;

    let buf: &mut [u64] =
        unsafe { core::slice::from_raw_parts_mut(pw.as_mut_ptr() as *mut u64, pw.len()) };
    let chg: u64 = field_to_u64(challenge);
    let (s0_raw, s1_raw, new_len) = reduce_and_evaluate::<Backend>(buf, chg);
    pw.truncate(new_len);

    let s0: F = u64_to_field(s0_raw);
    let s1: F = u64_to_field(s1_raw);
    Some(vec![s0, s1 - s0])
}

// ─── Extension field evaluate dispatch ──────────────────────────────────────

/// SIMD-accelerated pairwise evaluate for extension field elements.
///
/// Returns `Some((sum_even, sum_odd))` as extension field elements if
/// `EF` is a Goldilocks extension. Returns `None` otherwise.
///
/// The evaluate is pure addition (component-wise), so SIMD wins regardless
/// of extension degree.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_ext_evaluate<EF: Field>(evals: &[EF]) -> Option<(EF, EF)> {
    if !is_goldilocks_based::<EF>() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    let d = EF::extension_degree() as usize;

    if d == 1 {
        // Base field — use the optimized base evaluate
        let buf: &[u64] =
            unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, evals.len()) };
        let (s0, s1) = crate::simd_sumcheck::evaluate::evaluate_parallel::<Backend>(buf);
        return Some((u64_to_field(s0), u64_to_field(s1)));
    }

    // Extension field: view as flat u64 buffer and run ext_evaluate
    let n_u64 = evals.len() * d;
    let buf: &[u64] = unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, n_u64) };

    let (even_comps, odd_comps) =
        crate::simd_sumcheck::evaluate::ext_evaluate_parallel::<Backend>(buf, d);

    // Reconstruct extension field elements from component vectors
    let even: EF = unsafe { ext_components_to_field(&even_comps) };
    let odd: EF = unsafe { ext_components_to_field(&odd_comps) };

    Some((even, odd))
}

/// Reconstruct an extension field element from its raw u64 components.
///
/// # Safety
///
/// Components must be valid Montgomery-form u64 values and `F` must be
/// a Goldilocks extension with `size_of::<F>() == components.len() * 8`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
unsafe fn ext_components_to_field<F: Field>(components: &[u64]) -> F {
    debug_assert_eq!(core::mem::size_of::<F>(), components.len() * 8);
    let mut val = core::mem::MaybeUninit::<F>::uninit();
    core::ptr::copy_nonoverlapping(
        components.as_ptr(),
        val.as_mut_ptr() as *mut u64,
        components.len(),
    );
    val.assume_init()
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_ext_reduce<EF: Field>(evals: &mut Vec<EF>, challenge: EF) -> bool {
    if !is_goldilocks_based::<EF>() {
        return false;
    }

    let d = EF::extension_degree() as usize;

    if d == 1 {
        // Base field — use existing reduce
        return try_simd_reduce(evals, challenge);
    }

    if d == 2 {
        #[cfg(target_arch = "aarch64")]
        type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
        type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

        let chg_raw: [u64; 2] = unsafe {
            let ptr = &challenge as *const EF as *const u64;
            [*ptr, *ptr.add(1)]
        };
        let w = extract_nonresidue_ext2::<EF, Backend>();

        // In-place reduce: first half gets results, then truncate.
        let n_u64 = evals.len() * d;
        let buf: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(evals.as_mut_ptr() as *mut u64, n_u64) };
        crate::simd_sumcheck::reduce::ext2_reduce_in_place::<Backend>(buf, chg_raw, w);
        let new_len = evals.len() / 2;
        evals.truncate(new_len);
        return true;
    }

    if d == 3 {
        #[cfg(target_arch = "aarch64")]
        type Backend3 = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
        type Backend3 = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

        let chg_raw: [u64; 3] = unsafe {
            let ptr = &challenge as *const EF as *const u64;
            [*ptr, *ptr.add(1), *ptr.add(2)]
        };
        let w = extract_nonresidue_ext3::<EF, Backend3>();

        let n_u64 = evals.len() * d;
        let buf: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(evals.as_mut_ptr() as *mut u64, n_u64) };
        crate::simd_sumcheck::reduce::ext3_reduce_in_place::<Backend3>(buf, chg_raw, w);
        let new_len = evals.len() / 2;
        evals.truncate(new_len);
        return true;
    }

    // degree 4+: fall through to generic
    false
}

/// SIMD-accelerated degree-1 pairwise evaluate: returns `[s0, s1 - s0]`.
///
/// This is the coefficient sumcheck fast path for `degree() == 1` with a single
/// pairwise table and no tablewise tables — equivalent to the multilinear
/// `evaluate_parallel` kernel.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_evaluate_degree1<F: Field>(pw: &[F]) -> Option<Vec<F>> {
    if !is_goldilocks::<F>() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_sumcheck::evaluate::evaluate_parallel;

    let buf: &[u64] = unsafe { core::slice::from_raw_parts(pw.as_ptr() as *const u64, pw.len()) };
    let (s0_raw, s1_raw) = evaluate_parallel::<Backend>(buf);
    let s0: F = u64_to_field(s0_raw);
    let s1: F = u64_to_field(s1_raw);
    Some(vec![s0, s1 - s0])
}

// ─── Helpers: field ↔ u64 conversion ────────────────────────────────────────

/// Reinterpret a Montgomery-form `u64` as a field element.
///
/// Precondition: `F` is Goldilocks with `size_of::<F>() == 8`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
fn u64_to_field<F: Field>(raw: u64) -> F {
    debug_assert_eq!(core::mem::size_of::<F>(), 8);
    unsafe { core::mem::transmute_copy(&raw) }
}

/// Reinterpret a field element as its Montgomery-form `u64`.
///
/// Precondition: `F` is Goldilocks with `size_of::<F>() == 8`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
fn field_to_u64<F: Field>(val: F) -> u64 {
    debug_assert_eq!(core::mem::size_of::<F>(), 8);
    unsafe { core::mem::transmute_copy(&val) }
}

// ─── Public helpers for simd_ops ────────────────────────────────────────────

/// Check if `F` is a Goldilocks prime field (degree 1, size 8, matching modulus).
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
pub fn is_goldilocks_pub<F: Field>() -> bool {
    is_goldilocks::<F>()
}

/// Public wrapper — accepts base Goldilocks or any Goldilocks-based extension.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
pub fn is_goldilocks_based_pub<F: Field>() -> bool {
    is_goldilocks_based::<F>()
}

/// Reinterpret a Montgomery-form `u64` as a field element (public wrapper).
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
pub fn u64_to_field_pub<F: Field>(raw: u64) -> F {
    u64_to_field(raw)
}

/// Reinterpret a field element as its Montgomery-form `u64` (public wrapper).
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[inline(always)]
pub fn field_to_u64_pub<F: Field>(val: F) -> u64 {
    field_to_u64(val)
}
