//! SIMD auto-dispatch for the multilinear sumcheck protocol.
//!
//! When `BF == EF` and both are a Goldilocks field (p = 2^64 − 2^32 + 1)
//! stored as a single `u64` in Montgomery form, the sumcheck is transparently
//! routed to a NEON-accelerated backend.
//!
//! Detection uses [`Field::BasePrimeField::MODULUS`] from arkworks — no
//! concrete type names are referenced. After monomorphization the check
//! is constant-folded by LLVM, so the dead branch is eliminated entirely.

use ark_ff::Field;

use crate::multilinear::Sumcheck;
use crate::transcript::Transcript;

/// Goldilocks modulus: p = 2^64 − 2^32 + 1.
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
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn is_goldilocks<F: Field>() -> bool {
    use ark_ff::PrimeField; // for MODULUS on BasePrimeField

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

// ─── Auto-dispatch ──────────────────────────────────────────────────────────

/// Try to run the multilinear sumcheck on the SIMD backend.
///
/// Returns `Some(result)` if `BF == EF` is a recognised SIMD-accelerated
/// type (currently: Goldilocks). Returns `None` otherwise, letting the
/// caller fall through to the generic path.
///
/// # Safety invariant
///
/// When `is_goldilocks::<BF>()` is true we transmute `&[BF]` ↔ `&[u64]`.
/// This relies on `SmallFp<P>` (and `Fp64<MontBackend<_, 1>>`) having
/// the same in-memory layout as a bare `u64` — guaranteed in practice
/// because the only non-ZST field is `value: u64` (resp. `BigInt<1>([u64; 1])`).
/// A formal guarantee would require `#[repr(transparent)]` on those
/// structs or the `zerocopy` crate; until then the `size_of` check
/// provides a compile-time safety net.
#[cfg(target_arch = "aarch64")]
pub(crate) fn try_simd_dispatch<BF: Field, EF: Field + From<BF>>(
    evaluations: &mut [BF],
    transcript: &mut impl Transcript<EF>,
) -> Option<Sumcheck<EF>> {
    if !(is_goldilocks::<BF>() && is_goldilocks::<EF>()) {
        return None;
    }

    // ── Compile-time size sanity ────────────────────────────────────────
    // If the size check above somehow passed for a type whose layout
    // doesn't match u64, this assert will fire at compile time (const).
    assert!(
        core::mem::size_of::<BF>() == 8 && core::mem::size_of::<EF>() == 8,
        "Goldilocks dispatch: field element size must be 8 bytes"
    );

    use crate::simd_fields::goldilocks::neon::GoldilocksNeon;

    let n = evaluations.len();
    let num_rounds = n.trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<EF> = Vec::with_capacity(num_rounds);

    // Two strategies depending on input size:
    //
    // Small inputs (≤ HYBRID_THRESHOLD): all-SIMD path.
    //   SIMD evaluate (add) + SIMD in-place reduce (mul). The mul isn't
    //   truly vectorized on NEON (no 64×64→128), but for small arrays the
    //   overhead of cross-field reduce + Vec allocation costs more.
    //
    // Large inputs (> HYBRID_THRESHOLD): hybrid path.
    //   SIMD evaluate (add, genuine NEON speedup) + generic arkworks
    //   reduce (rayon-parallel Field ops outperform our scalar-fallback
    //   SIMD mul at scale).
    const HYBRID_THRESHOLD: usize = 1 << 18; // 262144 elements

    if n <= HYBRID_THRESHOLD {
        dispatch_all_simd::<BF, EF, GoldilocksNeon>(
            evaluations,
            transcript,
            num_rounds,
            &mut prover_messages,
            &mut verifier_messages,
        );
    } else {
        dispatch_hybrid::<BF, EF, GoldilocksNeon>(
            evaluations,
            transcript,
            num_rounds,
            &mut prover_messages,
            &mut verifier_messages,
        );
    }

    Some(Sumcheck {
        verifier_messages,
        prover_messages,
    })
}

/// All-SIMD path: evaluate + reduce both in raw u64 SIMD.
/// Best for small inputs where allocation overhead dominates.
#[cfg(target_arch = "aarch64")]
fn dispatch_all_simd<
    BF: Field,
    EF: Field + From<BF>,
    S: crate::simd_fields::SimdBaseField<Scalar = u64>,
>(
    evaluations: &[BF],
    transcript: &mut impl Transcript<EF>,
    num_rounds: usize,
    prover_messages: &mut Vec<(EF, EF)>,
    verifier_messages: &mut Vec<EF>,
) {
    use crate::simd_sumcheck::evaluate::evaluate_parallel;
    use crate::simd_sumcheck::reduce::reduce_in_place;

    // SAFETY: BF is Goldilocks, size_of == 8, layout-compatible with u64.
    let buf: &[u64] = unsafe {
        core::slice::from_raw_parts(evaluations.as_ptr() as *const u64, evaluations.len())
    };

    let mut current = buf.to_vec();
    let mut len = current.len();

    for round in 0..num_rounds {
        let (s0, s1) = evaluate_parallel::<S>(&current[..len]);

        let msg = (u64_to_field::<EF>(s0), u64_to_field::<EF>(s1));
        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        let chg_ef: EF = transcript.read();
        verifier_messages.push(chg_ef);

        if round < num_rounds - 1 {
            let chg: u64 = field_to_u64(chg_ef);
            len = reduce_in_place::<S>(&mut current[..len], chg);
        }
    }
}

/// Hybrid path: SIMD evaluate + generic arkworks reduce.
/// Best for large inputs where rayon-parallel Field reduce dominates.
#[cfg(target_arch = "aarch64")]
fn dispatch_hybrid<
    BF: Field,
    EF: Field + From<BF>,
    S: crate::simd_fields::SimdBaseField<Scalar = u64>,
>(
    evaluations: &[BF],
    transcript: &mut impl Transcript<EF>,
    num_rounds: usize,
    prover_messages: &mut Vec<(EF, EF)>,
    verifier_messages: &mut Vec<EF>,
) {
    use crate::multilinear::reductions::pairwise;
    use crate::simd_sumcheck::evaluate::evaluate_parallel;

    let n = evaluations.len();

    if num_rounds == 0 {
        return;
    }

    // ── Round 0: BF evaluate (SIMD) + cross-field reduce ──────────
    let buf: &[u64] = unsafe { core::slice::from_raw_parts(evaluations.as_ptr() as *const u64, n) };
    let (s0, s1) = evaluate_parallel::<S>(buf);

    let msg = (u64_to_field::<EF>(s0), u64_to_field::<EF>(s1));
    prover_messages.push(msg);
    transcript.write(msg.0);
    transcript.write(msg.1);

    let chg: EF = transcript.read();
    verifier_messages.push(chg);

    let mut ef_evals = pairwise::cross_field_reduce(evaluations, chg);

    // ── Rounds 1+: EF evaluate (SIMD) + EF reduce (generic) ──────
    for _ in 1..num_rounds {
        let buf: &[u64] =
            unsafe { core::slice::from_raw_parts(ef_evals.as_ptr() as *const u64, ef_evals.len()) };
        let (s0, s1) = evaluate_parallel::<S>(buf);

        let msg = (u64_to_field::<EF>(s0), u64_to_field::<EF>(s1));
        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        let chg: EF = transcript.read();
        verifier_messages.push(chg);

        pairwise::reduce_evaluations(&mut ef_evals, chg);
    }
}

// ─── Helpers: field ↔ u64 conversion ────────────────────────────────────────

/// Reinterpret a Montgomery-form `u64` as a field element.
///
/// Precondition: `F` is Goldilocks with `size_of::<F>() == 8`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn u64_to_field<F: Field>(raw: u64) -> F {
    debug_assert_eq!(core::mem::size_of::<F>(), 8);
    unsafe { core::mem::transmute_copy(&raw) }
}

/// Reinterpret a field element as its Montgomery-form `u64`.
///
/// Precondition: `F` is Goldilocks with `size_of::<F>() == 8`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn field_to_u64<F: Field>(val: F) -> u64 {
    debug_assert_eq!(core::mem::size_of::<F>(), 8);
    unsafe { core::mem::transmute_copy(&val) }
}
