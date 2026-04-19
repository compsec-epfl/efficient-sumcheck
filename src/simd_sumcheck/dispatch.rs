#![allow(dead_code)]
//! SIMD auto-dispatch for the multilinear sumcheck protocol.
//!
//! When `F` is a Goldilocks field (p = 2^64 − 2^32 + 1) stored as a single
//! `u64` in Montgomery form, the sumcheck is transparently routed to a
//! SIMD-accelerated backend:
//!
//! - **aarch64**: NEON backend (2-wide, scalar mul fallback)
//! - **x86_64 + AVX-512 IFMA**: AVX-512 backend (8-wide, true IFMA mul)
//!
//! Detection uses [`SumcheckField::_simd_field_config()`] — the arkworks
//! blanket impl returns the actual modulus, non-arkworks fields return
//! `None` by default (no SIMD). After monomorphization the check is
//! constant-folded by LLVM, so the dead branch is eliminated entirely.
//!
//! # Safety
//!
//! This module contains **no `unsafe` code**. All field ↔ `u64`
//! reinterpretation is delegated to the safe `SumcheckField` trait methods
//! (`_to_raw_u64`, `_from_raw_u64`, `_as_u64_slice`, `_as_u64_slice_mut`,
//! `_from_u64_components`), whose implementations centralize the necessary
//! `unsafe` in the arkworks blanket impl with full SAFETY documentation.

#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
use crate::field::SumcheckField;

/// Goldilocks modulus: p = 2^64 − 2^32 + 1.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Returns `true` when `F` is a Goldilocks prime field stored as a
/// single `u64` in Montgomery form.
///
/// Uses [`SumcheckField::_simd_field_config()`] for detection.
/// After monomorphization every operand is a compile-time constant,
/// so LLVM folds the entire function to `true` or `false`.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
#[inline(always)]
fn is_goldilocks<F: SumcheckField>() -> bool {
    if F::extension_degree() != 1 {
        return false;
    }
    match F::_simd_field_config() {
        Some(cfg) => cfg.modulus == GOLDILOCKS_P && cfg.element_bytes == 8,
        None => false,
    }
}

/// Returns `true` when `F` has Goldilocks as its base prime field,
/// regardless of extension degree. For degree-1 this is the same as
/// `is_goldilocks`. For degree 2, 3, etc., the element is `d` consecutive
/// `u64` values in Montgomery form.
///
/// After monomorphization, fully constant-folded by LLVM.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
#[inline(always)]
fn is_goldilocks_based<F: SumcheckField>() -> bool {
    match F::_simd_field_config() {
        Some(cfg) => {
            if cfg.modulus != GOLDILOCKS_P || cfg.element_bytes != 8 {
                return false;
            }
            let d = F::extension_degree() as usize;
            core::mem::size_of::<F>() == d * 8
        }
        None => false,
    }
}

// ─── Standalone SIMD reduce (Field-level API) ──────────────────────────────

/// SIMD-accelerated pairwise reduce on a `Vec<F>`.
///
/// If `F` is a recognised Goldilocks field, runs the SIMD reduce in-place
/// and truncates the vector. Otherwise returns `false` and the caller
/// should fall back to the generic path.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
pub(crate) fn try_simd_reduce<F: SumcheckField>(evals: &mut Vec<F>, challenge: F) -> bool {
    if !is_goldilocks::<F>() {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_sumcheck::reduce::reduce_in_place;

    let buf: &mut [u64] = F::_as_u64_slice_mut(evals.as_mut_slice());
    let chg: u64 = challenge._to_raw_u64();
    let new_len = reduce_in_place::<Backend>(buf, chg);
    evals.truncate(new_len);
    true
}

/// SIMD-accelerated MSB (half-split) reduce on a `Vec<F>`.
///
/// Like [`try_simd_reduce`] but uses the half-split layout:
/// `new[k] = v[k] + challenge * (v[k + L/2] − v[k])`.
/// Returns `false` for non-Goldilocks fields.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
pub(crate) fn try_simd_reduce_msb<F: SumcheckField>(evals: &mut Vec<F>, challenge: F) -> bool {
    if !is_goldilocks::<F>() {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_sumcheck::reduce::reduce_msb_in_place;

    let buf: &mut [u64] = F::_as_u64_slice_mut(evals.as_mut_slice());
    let chg: u64 = challenge._to_raw_u64();
    let new_len = reduce_msb_in_place::<Backend>(buf, chg);
    evals.truncate(new_len);
    true
}

// ─── SIMD degree-1 evaluate for coefficient sumcheck ────────────────────────

/// Fused SIMD reduce + degree-1 evaluate.
///
/// Reduces `pw` in-place and returns `[s0, s1 - s0]` for the next round,
/// computed in a single data pass via `reduce_and_evaluate`.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
pub(crate) fn try_simd_fused_reduce_evaluate_degree1<F: SumcheckField>(
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

    let buf: &mut [u64] = F::_as_u64_slice_mut(pw.as_mut_slice());
    let chg: u64 = challenge._to_raw_u64();
    let (s0_raw, s1_raw, new_len) = reduce_and_evaluate::<Backend>(buf, chg);
    pw.truncate(new_len);

    let s0: F = F::_from_raw_u64(s0_raw);
    let s1: F = F::_from_raw_u64(s1_raw);
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
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
pub(crate) fn try_simd_ext_evaluate<EF: SumcheckField>(evals: &[EF]) -> Option<(EF, EF)> {
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
        let buf: &[u64] = EF::_as_u64_slice(evals);
        let (s0, s1) = crate::simd_sumcheck::evaluate::evaluate_parallel::<Backend>(buf);
        return Some((EF::_from_raw_u64(s0), EF::_from_raw_u64(s1)));
    }

    // Extension field: view as flat u64 buffer and run ext_evaluate
    let buf: &[u64] = EF::_as_u64_slice(evals);

    let (even_comps, odd_comps) =
        crate::simd_sumcheck::evaluate::ext_evaluate_parallel::<Backend>(buf, d);

    let even: EF = EF::_from_u64_components(&even_comps);
    let odd: EF = EF::_from_u64_components(&odd_comps);

    Some((even, odd))
}

/// SIMD-accelerated degree-1 pairwise evaluate: returns `[s0, s1 - s0]`.
///
/// This is the coefficient sumcheck fast path for `degree() == 1` with a single
/// pairwise table and no tablewise tables — equivalent to the multilinear
/// `evaluate_parallel` kernel.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
pub(crate) fn try_simd_evaluate_degree1<F: SumcheckField>(pw: &[F]) -> Option<Vec<F>> {
    if !is_goldilocks::<F>() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_sumcheck::evaluate::evaluate_parallel;

    let buf: &[u64] = F::_as_u64_slice(pw);
    let (s0_raw, s1_raw) = evaluate_parallel::<Backend>(buf);
    let s0: F = F::_from_raw_u64(s0_raw);
    let s1: F = F::_from_raw_u64(s1_raw);
    Some(vec![s0, s1 - s0])
}

// ─── Public helpers ────────────────────────────────────────────────────────

/// Check if `F` is a Goldilocks prime field (degree 1, size 8, matching modulus).
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
#[inline(always)]
pub fn is_goldilocks_pub<F: SumcheckField>() -> bool {
    is_goldilocks::<F>()
}
