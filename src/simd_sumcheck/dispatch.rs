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

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
use crate::multilinear::Sumcheck;
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
use crate::transcript::Transcript;

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
pub(crate) fn extract_nonresidue_ext2<EF: Field, S: crate::simd_fields::SimdBaseField<Scalar = u64>>() -> u64 {
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
pub(crate) fn extract_nonresidue_ext3<EF: Field, S: crate::simd_fields::SimdBaseField<Scalar = u64>>() -> u64 {
    let one_x = unsafe {
        let mut tmp = [0u64; 3];
        tmp[1] = S::ONE;
        let one_x: EF = core::mem::transmute_copy(&tmp);
        one_x
    };
    let nr = one_x * one_x * one_x;
    unsafe { *((&nr) as *const EF as *const u64) }
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
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_dispatch<BF, EF, T, H>(
    evaluations: &mut [BF],
    transcript: &mut T,
    hook: &mut H,
) -> Option<Sumcheck<EF>>
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
{
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

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    let n = evaluations.len();
    let num_rounds = n.trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<EF> = Vec::with_capacity(num_rounds);

    // Two strategies depending on input size:
    //
    // Small inputs (≤ HYBRID_THRESHOLD): all-SIMD path.
    //   SIMD evaluate (add) + SIMD in-place reduce (mul).
    //
    // Large inputs (> HYBRID_THRESHOLD): hybrid path.
    //   SIMD evaluate (add) + generic arkworks reduce (rayon-parallel).
    //
    // The threshold is architecture-dependent:
    //
    // NEON: mul falls back to scalar (no 64×64→128), so the hybrid path
    //   (in-place generic reduce) wins at scale. Threshold at 2^18.
    //
    // AVX-512 IFMA: mul is truly 8-wide vectorized, so the all-SIMD path
    //   stays competitive longer. At very large sizes memory bandwidth
    //   dominates and the hybrid path (which avoids extra allocation)
    //   catches up. Threshold at 2^20 balances SIMD reduce wins with
    //   memory traffic.
    #[cfg(target_arch = "aarch64")]
    const HYBRID_THRESHOLD: usize = 1 << 18;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    const HYBRID_THRESHOLD: usize = 1 << 30;

    let final_evaluation = if n <= HYBRID_THRESHOLD {
        dispatch_all_simd::<BF, EF, T, H, Backend>(
            evaluations,
            transcript,
            hook,
            num_rounds,
            &mut prover_messages,
            &mut verifier_messages,
        )
    } else {
        dispatch_hybrid::<BF, EF, T, H, Backend>(
            evaluations,
            transcript,
            hook,
            num_rounds,
            &mut prover_messages,
            &mut verifier_messages,
        )
    };

    Some(Sumcheck {
        verifier_messages,
        prover_messages,
        final_evaluation,
    })
}

/// Try to run the multilinear sumcheck on the SIMD backend for extension fields.
///
/// Handles the case where BF == EF and EF is a Goldilocks extension (degree 2 or 3).
/// Uses SoA (Struct-of-Arrays) layout: converts AoS to SoA once at entry, then
/// all rounds operate on contiguous component arrays. This eliminates all shuffle
/// overhead (permutex2var, gather/scatter) from the AoS reduce path.
///
/// Evaluate becomes per-component `evaluate_parallel` (fully SIMD, ~6x over generic).
/// Reduce uses contiguous loads with `load_deinterleaved` (no shuffles).
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
#[allow(dead_code)] // Used on AVX-512; on NEON, generic path with rayon is faster
pub(crate) fn try_simd_ext_dispatch<BF, EF, T, H>(
    evaluations: &mut [BF],
    transcript: &mut T,
    hook: &mut H,
) -> Option<Sumcheck<EF>>
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
{
    if !is_goldilocks_based::<BF>() {
        return None;
    }

    let d = BF::extension_degree() as usize;
    if !(2..=3).contains(&d) {
        return None;
    }

    // BF must be the same as EF (both are ext fields with same layout)
    if core::mem::size_of::<BF>() != core::mem::size_of::<EF>() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    let n = evaluations.len();
    let num_rounds = n.trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<EF> = Vec::with_capacity(num_rounds);
    let mut final_evaluation = EF::ZERO;

    // View evaluations as flat u64 buffer
    let n_u64 = n * d;
    let src: &[u64] =
        unsafe { core::slice::from_raw_parts(evaluations.as_ptr() as *const u64, n_u64) };

    // Above this input size, switch to rayon-parallel SoA reduce. Below it,
    // the in-place single-threaded kernel wins (thread scheduling overhead
    // dominates the small chunk work).
    const EXT_PARALLEL_THRESHOLD: usize = 1 << 17;

    if d == 2 {
        let w = extract_nonresidue_ext2::<EF, Backend>();

        // Convert AoS → SoA once (one-time O(n) cost, eliminates per-round shuffles)
        let (mut c0, mut c1) = aos_to_soa_ext2(src);
        let mut len = n; // number of extension elements

        // Scratch for parallel ping-pong (read from c*, write to scratch_*, swap).
        // Size n/2 is enough for the first parallel round; subsequent rounds write
        // smaller outputs.
        let use_parallel = n > EXT_PARALLEL_THRESHOLD;
        let mut scratch_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut scratch_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };

        // Fused reduce+evaluate: rounds 1+ get evaluate results from the prior
        // round's fused kernel, eliminating one full data pass per round.
        let mut pending_eval: Option<([u64; 2], [u64; 2])> = None;

        for round in 0..num_rounds {
            let (even_comps, odd_comps) = pending_eval.unwrap_or_else(|| {
                use crate::simd_sumcheck::evaluate::evaluate_parallel;
                let (e0, o0) = evaluate_parallel::<Backend>(&c0[..len]);
                let (e1, o1) = evaluate_parallel::<Backend>(&c1[..len]);
                ([e0, e1], [o0, o1])
            });

            let even: EF = unsafe { ext_components_to_field(&even_comps) };
            let odd: EF = unsafe { ext_components_to_field(&odd_comps) };
            let msg = (even, odd);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            hook(round, transcript);

            let chg: EF = transcript.read();
            verifier_messages.push(chg);

            if round < num_rounds - 1 {
                let chg_raw: [u64; 2] = unsafe {
                    let ptr = &chg as *const EF as *const u64;
                    [*ptr, *ptr.add(1)]
                };
                if len > EXT_PARALLEL_THRESHOLD {
                    let new_len = len / 2;
                    let (next_even, next_odd) =
                        crate::simd_sumcheck::reduce::ext2_soa_reduce_and_evaluate_parallel::<Backend>(
                            &c0[..len], &c1[..len],
                            &mut scratch_c0[..new_len], &mut scratch_c1[..new_len],
                            chg_raw, w,
                        );
                    core::mem::swap(&mut c0, &mut scratch_c0);
                    core::mem::swap(&mut c1, &mut scratch_c1);
                    len = new_len;
                    pending_eval = Some((next_even, next_odd));
                } else {
                    let (next_even, next_odd, new_len) =
                        crate::simd_sumcheck::reduce::ext2_soa_reduce_and_evaluate::<Backend>(
                            &mut c0[..len], &mut c1[..len], chg_raw, w,
                        );
                    len = new_len;
                    pending_eval = Some((next_even, next_odd));
                }
            } else {
                // Last round: fold the surviving pair with the final challenge
                // (in EF arithmetic — independent of `w`).
                debug_assert_eq!(len, 2);
                let v0: EF = unsafe { ext_components_to_field(&[c0[0], c1[0]]) };
                let v1: EF = unsafe { ext_components_to_field(&[c0[1], c1[1]]) };
                final_evaluation = v0 + chg * (v1 - v0);
            }
        }
    } else {
        // d == 3
        let w = extract_nonresidue_ext3::<EF, Backend>();

        let (mut c0, mut c1, mut c2) = aos_to_soa_ext3(src);
        let mut len = n;
        let use_parallel = n > EXT_PARALLEL_THRESHOLD;
        let mut scratch_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut scratch_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut scratch_c2: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut pending_eval: Option<([u64; 3], [u64; 3])> = None;

        for round in 0..num_rounds {
            let (even_comps, odd_comps) = pending_eval.unwrap_or_else(|| {
                use crate::simd_sumcheck::evaluate::evaluate_parallel;
                let (e0, o0) = evaluate_parallel::<Backend>(&c0[..len]);
                let (e1, o1) = evaluate_parallel::<Backend>(&c1[..len]);
                let (e2, o2) = evaluate_parallel::<Backend>(&c2[..len]);
                ([e0, e1, e2], [o0, o1, o2])
            });

            let even: EF = unsafe { ext_components_to_field(&even_comps) };
            let odd: EF = unsafe { ext_components_to_field(&odd_comps) };
            let msg = (even, odd);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            hook(round, transcript);

            let chg: EF = transcript.read();
            verifier_messages.push(chg);

            if round < num_rounds - 1 {
                let chg_raw: [u64; 3] = unsafe {
                    let ptr = &chg as *const EF as *const u64;
                    [*ptr, *ptr.add(1), *ptr.add(2)]
                };
                if len > EXT_PARALLEL_THRESHOLD {
                    let new_len = len / 2;
                    let (next_even, next_odd) =
                        crate::simd_sumcheck::reduce::ext3_soa_reduce_and_evaluate_parallel::<Backend>(
                            &c0[..len], &c1[..len], &c2[..len],
                            &mut scratch_c0[..new_len], &mut scratch_c1[..new_len], &mut scratch_c2[..new_len],
                            chg_raw, w,
                        );
                    core::mem::swap(&mut c0, &mut scratch_c0);
                    core::mem::swap(&mut c1, &mut scratch_c1);
                    core::mem::swap(&mut c2, &mut scratch_c2);
                    len = new_len;
                    pending_eval = Some((next_even, next_odd));
                } else {
                    let (next_even, next_odd, new_len) =
                        crate::simd_sumcheck::reduce::ext3_soa_reduce_and_evaluate::<Backend>(
                            &mut c0[..len], &mut c1[..len], &mut c2[..len], chg_raw, w,
                        );
                    len = new_len;
                    pending_eval = Some((next_even, next_odd));
                }
            } else {
                debug_assert_eq!(len, 2);
                let v0: EF = unsafe { ext_components_to_field(&[c0[0], c1[0], c2[0]]) };
                let v1: EF = unsafe { ext_components_to_field(&[c0[1], c1[1], c2[1]]) };
                final_evaluation = v0 + chg * (v1 - v0);
            }
        }
    }

    Some(Sumcheck {
        verifier_messages,
        prover_messages,
        final_evaluation,
    })
}

/// All-SIMD path: evaluate + reduce both in raw u64 SIMD.
/// Best for small-to-medium inputs where SIMD reduce beats generic.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
fn dispatch_all_simd<BF, EF, T, H, S>(
    evaluations: &mut [BF],
    transcript: &mut T,
    hook: &mut H,
    num_rounds: usize,
    prover_messages: &mut Vec<(EF, EF)>,
    verifier_messages: &mut Vec<EF>,
) -> EF
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
    S: crate::simd_fields::SimdBaseField<Scalar = u64>,
{
    use crate::simd_sumcheck::evaluate::evaluate_parallel;
    use crate::simd_sumcheck::reduce::{reduce_and_evaluate, reduce_in_place};

    // SAFETY: BF is Goldilocks, size_of == 8, layout-compatible with u64.
    // Work in-place on the evaluation buffer to avoid allocation overhead.
    let current: &mut [u64] = unsafe {
        core::slice::from_raw_parts_mut(evaluations.as_mut_ptr() as *mut u64, evaluations.len())
    };

    let mut len = current.len();

    // Fused reduce+evaluate eliminates one data pass per round.
    // Only beneficial when data exceeds L2 cache (~2 MB = ~2^18 u64s).
    const FUSE_THRESHOLD: usize = 1 << 20;

    let mut pending_eval: Option<(u64, u64)> = None;

    for round in 0..num_rounds {
        let (s0, s1) = pending_eval.unwrap_or_else(|| evaluate_parallel::<S>(&current[..len]));

        let msg = (u64_to_field::<EF>(s0), u64_to_field::<EF>(s1));
        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        hook(round, transcript);

        let chg_ef: EF = transcript.read();
        verifier_messages.push(chg_ef);

        if round < num_rounds - 1 {
            let chg: u64 = field_to_u64(chg_ef);
            if len > FUSE_THRESHOLD {
                let (ns0, ns1, new_len) = reduce_and_evaluate::<S>(&mut current[..len], chg);
                len = new_len;
                pending_eval = Some((ns0, ns1));
            } else {
                len = reduce_in_place::<S>(&mut current[..len], chg);
                pending_eval = None;
            }
        } else if num_rounds > 0 {
            // Last round: fold the surviving pair with the final challenge.
            debug_assert_eq!(len, 2);
            let v0: EF = u64_to_field(current[0]);
            let v1: EF = u64_to_field(current[1]);
            return v0 + chg_ef * (v1 - v0);
        }
    }
    EF::ZERO
}

/// Hybrid path: SIMD evaluate + generic arkworks reduce.
/// Best for large inputs where rayon-parallel Field reduce dominates.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
fn dispatch_hybrid<BF, EF, T, H, S>(
    evaluations: &[BF],
    transcript: &mut T,
    hook: &mut H,
    num_rounds: usize,
    prover_messages: &mut Vec<(EF, EF)>,
    verifier_messages: &mut Vec<EF>,
) -> EF
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
    S: crate::simd_fields::SimdBaseField<Scalar = u64>,
{
    use crate::multilinear::reductions::pairwise;
    use crate::simd_sumcheck::evaluate::evaluate_parallel;

    let n = evaluations.len();

    if num_rounds == 0 {
        return EF::ZERO;
    }

    // ── Round 0: BF evaluate (SIMD) + cross-field reduce ──────────
    let buf: &[u64] = unsafe { core::slice::from_raw_parts(evaluations.as_ptr() as *const u64, n) };
    let (s0, s1) = evaluate_parallel::<S>(buf);

    let msg = (u64_to_field::<EF>(s0), u64_to_field::<EF>(s1));
    prover_messages.push(msg);
    transcript.write(msg.0);
    transcript.write(msg.1);

    hook(0, transcript);

    let chg: EF = transcript.read();
    verifier_messages.push(chg);

    let mut ef_evals = pairwise::cross_field_reduce(evaluations, chg);

    // ── Rounds 1+: EF evaluate (SIMD) + EF reduce (generic) ──────
    for round in 1..num_rounds {
        let buf: &[u64] =
            unsafe { core::slice::from_raw_parts(ef_evals.as_ptr() as *const u64, ef_evals.len()) };
        let (s0, s1) = evaluate_parallel::<S>(buf);

        let msg = (u64_to_field::<EF>(s0), u64_to_field::<EF>(s1));
        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        hook(round, transcript);

        let chg: EF = transcript.read();
        verifier_messages.push(chg);

        pairwise::reduce_evaluations(&mut ef_evals, chg);
    }

    debug_assert_eq!(ef_evals.len(), 1);
    ef_evals[0]
}

// ─── Inner product dispatch ─────────────────────────────────────────────────

/// Try to run the inner product sumcheck on the SIMD backend.
///
/// Same safety invariant as [`try_simd_dispatch`].
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_product_dispatch<BF, EF, T, H>(
    f: &mut [BF],
    g: &mut [BF],
    transcript: &mut T,
    hook: &mut H,
) -> Option<crate::multilinear_product::ProductSumcheck<EF>>
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
{
    if !(is_goldilocks::<BF>() && is_goldilocks::<EF>()) {
        return None;
    }

    assert!(
        core::mem::size_of::<BF>() == 8 && core::mem::size_of::<EF>() == 8,
        "Goldilocks dispatch: field element size must be 8 bytes"
    );

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_sumcheck::evaluate::product_evaluate_parallel;
    use crate::simd_sumcheck::reduce::reduce_both_in_place;

    let n = f.len();
    let num_rounds = n.trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<EF> = Vec::with_capacity(num_rounds);
    let mut final_evaluations = (EF::ZERO, EF::ZERO);

    if num_rounds > 0 {
        let f_raw: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(f.as_mut_ptr() as *mut u64, n) };
        let g_raw: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(g.as_mut_ptr() as *mut u64, n) };

        let mut len = n;

        for round in 0..num_rounds {
            let (a, b) = product_evaluate_parallel::<Backend>(&f_raw[..len], &g_raw[..len]);

            let msg = (u64_to_field::<EF>(a), u64_to_field::<EF>(b));
            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            hook(round, transcript);

            let chg_ef: EF = transcript.read();
            verifier_messages.push(chg_ef);

            if round < num_rounds - 1 {
                let chg: u64 = field_to_u64(chg_ef);
                // Reduce both f and g in one interleaved pass (saves one full data read)
                len = reduce_both_in_place::<Backend>(&mut f_raw[..len], &mut g_raw[..len], chg);
            } else {
                // Last round: compute the final folded values using the last
                // challenge. The loop guard skips the in-place reduce, so
                // f_raw[0..2] and g_raw[0..2] still hold the surviving pair.
                debug_assert_eq!(len, 2);
                let f0: EF = u64_to_field(f_raw[0]);
                let f1: EF = u64_to_field(f_raw[1]);
                let g0: EF = u64_to_field(g_raw[0]);
                let g1: EF = u64_to_field(g_raw[1]);
                final_evaluations = (f0 + chg_ef * (f1 - f0), g0 + chg_ef * (g1 - g0));
            }
        }
    }

    Some(crate::multilinear_product::ProductSumcheck {
        verifier_messages,
        prover_messages,
        final_evaluations,
    })
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

/// SIMD-accelerated extension field reduce on `Vec<EF>`.
///
/// For degree-2 Goldilocks extensions: uses `ext2_reduce_in_place` with
/// specialized Karatsuba multiply. Returns `true` if handled.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
/// Fused extension reduce + next-round evaluate.
///
/// Reduces `evals` in-place and returns `Some((next_even, next_odd))` for the
/// next round's prover message. Returns `None` for unsupported fields.
/// This eliminates one full data pass per round.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_ext_fused_reduce_evaluate<EF: Field>(
    evals: &mut Vec<EF>,
    challenge: EF,
) -> Option<(EF, EF)> {
    if !is_goldilocks_based::<EF>() {
        return None;
    }

    let d = EF::extension_degree() as usize;

    if d == 1 {
        // Base field: use existing fused reduce_and_evaluate
        #[cfg(target_arch = "aarch64")]
        type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
        type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

        let buf: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(evals.as_mut_ptr() as *mut u64, evals.len()) };
        let chg: u64 = field_to_u64(challenge);
        let (s0, s1, new_len) =
            crate::simd_sumcheck::reduce::reduce_and_evaluate::<Backend>(buf, chg);
        evals.truncate(new_len);
        return Some((u64_to_field(s0), u64_to_field(s1)));
    }

    #[cfg(target_arch = "aarch64")]
    {
        if d == 2 {
            let n_u64 = evals.len() * d;
            let buf: &mut [u64] =
                unsafe { core::slice::from_raw_parts_mut(evals.as_mut_ptr() as *mut u64, n_u64) };

            let chg_raw: [u64; 2] = unsafe {
                let ptr = &challenge as *const EF as *const u64;
                [*ptr, *ptr.add(1)]
            };

            // Extract nonresidue
            let w = extract_ext2_nonresidue::<EF>();

            let (even_comps, odd_comps, new_len_u64) =
                crate::simd_sumcheck::reduce::ext2_reduce_and_evaluate(buf, chg_raw, w);
            evals.truncate(new_len_u64 / d);

            let even: EF = unsafe { ext_components_to_field(&even_comps) };
            let odd: EF = unsafe { ext_components_to_field(&odd_comps) };
            return Some((even, odd));
        }

        if d == 3 {
            let n_u64 = evals.len() * d;
            let buf: &mut [u64] =
                unsafe { core::slice::from_raw_parts_mut(evals.as_mut_ptr() as *mut u64, n_u64) };

            let chg_raw: [u64; 3] = unsafe {
                let ptr = &challenge as *const EF as *const u64;
                [*ptr, *ptr.add(1), *ptr.add(2)]
            };

            let w = extract_ext2_nonresidue::<EF>(); // same trick works for ext3

            let (even_comps, odd_comps, new_len_u64) =
                crate::simd_sumcheck::reduce::ext3_reduce_and_evaluate(buf, chg_raw, w);
            evals.truncate(new_len_u64 / d);

            let even: EF = unsafe { ext_components_to_field(&even_comps) };
            let odd: EF = unsafe { ext_components_to_field(&odd_comps) };
            return Some((even, odd));
        }
    }

    None
}

/// Extract the nonresidue w from an extension field at runtime.
/// Computes (0, 1, 0...) * (0, 1, 0...) = (w, 0, 0...) and extracts the first component.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
fn extract_ext2_nonresidue<EF: Field>() -> u64 {
    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    use crate::simd_fields::SimdBaseField;

    let d = EF::extension_degree() as usize;
    let one_x: EF = unsafe {
        let mut tmp = vec![0u64; d];
        tmp[1] = Backend::ONE;
        let mut val = core::mem::MaybeUninit::<EF>::uninit();
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), val.as_mut_ptr() as *mut u64, d);
        val.assume_init()
    };
    let nr = one_x * one_x;
    unsafe { *((&nr) as *const EF as *const u64) }
}

#[allow(dead_code)]
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

// ─── AoS → SoA conversion ──────────────────────────────────────────────────

/// Convert AoS ext2 layout to SoA: [e0_c0, e0_c1, e1_c0, e1_c1, ...] → (c0[], c1[])
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn aos_to_soa_ext2(src: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let n = src.len() / 2;
    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    for i in 0..n {
        c0.push(src[2 * i]);
        c1.push(src[2 * i + 1]);
    }
    (c0, c1)
}

/// Convert AoS ext3 layout to SoA: [e0_c0, e0_c1, e0_c2, e1_c0, ...] → (c0[], c1[], c2[])
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn aos_to_soa_ext3(src: &[u64]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = src.len() / 3;
    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        c0.push(src[3 * i]);
        c1.push(src[3 * i + 1]);
        c2.push(src[3 * i + 2]);
    }
    (c0, c1, c2)
}

// ─── Inner product extension dispatch ──────────────────────────────────────

/// Try to run the inner product sumcheck on the SIMD backend for extension fields.
///
/// Handles BF == EF == Goldilocks ext2 (degree-2 extension).
/// Uses SoA layout for both f and g, with SIMD product evaluate + SoA reduce.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_ext_product_dispatch<BF, EF, T, H>(
    f: &mut [BF],
    g: &mut [BF],
    transcript: &mut T,
    hook: &mut H,
) -> Option<crate::multilinear_product::ProductSumcheck<EF>>
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
{
    if !is_goldilocks_based::<BF>() {
        return None;
    }

    let d = BF::extension_degree() as usize;
    if !(2..=3).contains(&d) {
        return None;
    }

    if core::mem::size_of::<BF>() != core::mem::size_of::<EF>() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    let n = f.len();
    let num_rounds = n.trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<EF> = Vec::with_capacity(num_rounds);
    let mut final_evaluations = (EF::ZERO, EF::ZERO);

    // Convert both f and g from AoS → SoA
    let f_u64: &[u64] =
        unsafe { core::slice::from_raw_parts(f.as_ptr() as *const u64, n * d) };
    let g_u64: &[u64] =
        unsafe { core::slice::from_raw_parts(g.as_ptr() as *const u64, n * d) };

    const EXT_PARALLEL_THRESHOLD: usize = 1 << 17;

    // NOTE on fusion: unlike the non-product SoA dispatch, we don't use a
    // pending_eval optimization here. The product evaluate requires Σ f'[2m']·g'[2m']
    // on the *reduced* values, which needs lane-deinterleaving + Karatsuba across
    // the two halves of each SIMD register — more complex than the non-product
    // case (which just sums even/odd lanes). Call product_evaluate per round
    // and reduce separately; the correct fusion is a future optimization.
    if d == 2 {
        let w = extract_nonresidue_ext2::<EF, Backend>();

        let (mut f_c0, mut f_c1) = aos_to_soa_ext2(f_u64);
        let (mut g_c0, mut g_c1) = aos_to_soa_ext2(g_u64);
        let mut len = n;

        let use_parallel = n > EXT_PARALLEL_THRESHOLD;
        let mut sf_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sf_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };

        for round in 0..num_rounds {
            let (a_raw, b_raw) =
                crate::simd_sumcheck::reduce::ext2_soa_product_evaluate::<Backend>(
                    &f_c0[..len], &f_c1[..len],
                    &g_c0[..len], &g_c1[..len],
                    w,
                );

            let a: EF = unsafe { ext_components_to_field(&a_raw) };
            let b: EF = unsafe { ext_components_to_field(&b_raw) };
            let msg = (a, b);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            hook(round, transcript);

            let chg: EF = transcript.read();
            verifier_messages.push(chg);

            if round < num_rounds - 1 {
                let chg_raw: [u64; 2] = unsafe {
                    let ptr = &chg as *const EF as *const u64;
                    [*ptr, *ptr.add(1)]
                };
                if len > EXT_PARALLEL_THRESHOLD {
                    let new_len = len / 2;
                    // Discard the (wrong) evaluate return; we recompute it at next
                    // round's start.
                    let _ = crate::simd_sumcheck::reduce::ext2_soa_product_reduce_and_evaluate_parallel::<Backend>(
                        &f_c0[..len], &f_c1[..len],
                        &g_c0[..len], &g_c1[..len],
                        &mut sf_c0[..new_len], &mut sf_c1[..new_len],
                        &mut sg_c0[..new_len], &mut sg_c1[..new_len],
                        chg_raw, w,
                    );
                    core::mem::swap(&mut f_c0, &mut sf_c0);
                    core::mem::swap(&mut f_c1, &mut sf_c1);
                    core::mem::swap(&mut g_c0, &mut sg_c0);
                    core::mem::swap(&mut g_c1, &mut sg_c1);
                    len = new_len;
                } else {
                    let (_, _, new_len) =
                        crate::simd_sumcheck::reduce::ext2_soa_product_reduce_and_evaluate::<Backend>(
                            &mut f_c0[..len], &mut f_c1[..len],
                            &mut g_c0[..len], &mut g_c1[..len],
                            chg_raw, w,
                        );
                    len = new_len;
                }
            } else {
                // Last round: compute final folded values from the surviving
                // pair using EF arithmetic.
                debug_assert_eq!(len, 2);
                let f0: EF = unsafe { ext_components_to_field(&[f_c0[0], f_c1[0]]) };
                let f1: EF = unsafe { ext_components_to_field(&[f_c0[1], f_c1[1]]) };
                let g0: EF = unsafe { ext_components_to_field(&[g_c0[0], g_c1[0]]) };
                let g1: EF = unsafe { ext_components_to_field(&[g_c0[1], g_c1[1]]) };
                final_evaluations = (f0 + chg * (f1 - f0), g0 + chg * (g1 - g0));
            }
        }
    } else {
        // d == 3
        let w = extract_nonresidue_ext3::<EF, Backend>();

        let (mut f_c0, mut f_c1, mut f_c2) = aos_to_soa_ext3(f_u64);
        let (mut g_c0, mut g_c1, mut g_c2) = aos_to_soa_ext3(g_u64);
        let mut len = n;

        let use_parallel = n > EXT_PARALLEL_THRESHOLD;
        let mut sf_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sf_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sf_c2: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c2: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };

        for round in 0..num_rounds {
            let (a_raw, b_raw) =
                crate::simd_sumcheck::reduce::ext3_soa_product_evaluate::<Backend>(
                    &f_c0[..len], &f_c1[..len], &f_c2[..len],
                    &g_c0[..len], &g_c1[..len], &g_c2[..len],
                    w,
                );

            let a: EF = unsafe { ext_components_to_field(&a_raw) };
            let b: EF = unsafe { ext_components_to_field(&b_raw) };
            let msg = (a, b);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            hook(round, transcript);

            let chg: EF = transcript.read();
            verifier_messages.push(chg);

            if round < num_rounds - 1 {
                let chg_raw: [u64; 3] = unsafe {
                    let ptr = &chg as *const EF as *const u64;
                    [*ptr, *ptr.add(1), *ptr.add(2)]
                };
                if len > EXT_PARALLEL_THRESHOLD {
                    let new_len = len / 2;
                    let _ = crate::simd_sumcheck::reduce::ext3_soa_product_reduce_and_evaluate_parallel::<Backend>(
                        &f_c0[..len], &f_c1[..len], &f_c2[..len],
                        &g_c0[..len], &g_c1[..len], &g_c2[..len],
                        &mut sf_c0[..new_len], &mut sf_c1[..new_len], &mut sf_c2[..new_len],
                        &mut sg_c0[..new_len], &mut sg_c1[..new_len], &mut sg_c2[..new_len],
                        chg_raw, w,
                    );
                    core::mem::swap(&mut f_c0, &mut sf_c0);
                    core::mem::swap(&mut f_c1, &mut sf_c1);
                    core::mem::swap(&mut f_c2, &mut sf_c2);
                    core::mem::swap(&mut g_c0, &mut sg_c0);
                    core::mem::swap(&mut g_c1, &mut sg_c1);
                    core::mem::swap(&mut g_c2, &mut sg_c2);
                    len = new_len;
                } else {
                    let (_, _, new_len) =
                        crate::simd_sumcheck::reduce::ext3_soa_product_reduce_and_evaluate::<Backend>(
                            &mut f_c0[..len], &mut f_c1[..len], &mut f_c2[..len],
                            &mut g_c0[..len], &mut g_c1[..len], &mut g_c2[..len],
                            chg_raw, w,
                        );
                    len = new_len;
                }
            } else {
                debug_assert_eq!(len, 2);
                let f0: EF =
                    unsafe { ext_components_to_field(&[f_c0[0], f_c1[0], f_c2[0]]) };
                let f1: EF =
                    unsafe { ext_components_to_field(&[f_c0[1], f_c1[1], f_c2[1]]) };
                let g0: EF =
                    unsafe { ext_components_to_field(&[g_c0[0], g_c1[0], g_c2[0]]) };
                let g1: EF =
                    unsafe { ext_components_to_field(&[g_c0[1], g_c1[1], g_c2[1]]) };
                final_evaluations = (f0 + chg * (f1 - f0), g0 + chg * (g1 - g0));
            }
        }
    }

    Some(crate::multilinear_product::ProductSumcheck {
        verifier_messages,
        prover_messages,
        final_evaluations,
    })
}

// ─── Partial IP extension dispatch (SoA-persistent across rounds) ──────────

/// Run `max_rounds` rounds of inner-product sumcheck over a Goldilocks ext2
/// or ext3 field, keeping SoA state across rounds (one AoS→SoA at entry, one
/// SoA→AoS at exit — `max_rounds − 1` round-trips avoided vs the per-round
/// AoS↔SoA `pairwise_product_sum` + `fold_both` loop).
///
/// On success, truncates `f` and `g` to the folded length (`f.len() >> max_rounds`).
/// Returns `None` if `F` is not Goldilocks ext2 or ext3.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
pub(crate) fn try_simd_ext_product_partial_dispatch<F, T, H>(
    f: &mut Vec<F>,
    g: &mut Vec<F>,
    transcript: &mut T,
    max_rounds: usize,
    hook: &mut H,
) -> Option<crate::multilinear_product::ProductSumcheck<F>>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    if !is_goldilocks_based::<F>() {
        return None;
    }
    let d = F::extension_degree() as usize;
    if !(2..=3).contains(&d) {
        return None;
    }
    if core::mem::size_of::<F>() != d * core::mem::size_of::<u64>() {
        return None;
    }

    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    let n = f.len();
    debug_assert_eq!(n, g.len());
    let total_rounds = n.trailing_zeros() as usize;
    assert!(max_rounds <= total_rounds);

    let mut prover_messages: Vec<(F, F)> = Vec::with_capacity(max_rounds);
    let mut verifier_messages: Vec<F> = Vec::with_capacity(max_rounds);

    let f_u64: &[u64] =
        unsafe { core::slice::from_raw_parts(f.as_ptr() as *const u64, n * d) };
    let g_u64: &[u64] =
        unsafe { core::slice::from_raw_parts(g.as_ptr() as *const u64, n * d) };

    const EXT_PARALLEL_THRESHOLD: usize = 1 << 17;

    if d == 2 {
        let w = extract_nonresidue_ext2::<F, Backend>();

        let (mut f_c0, mut f_c1) = aos_to_soa_ext2(f_u64);
        let (mut g_c0, mut g_c1) = aos_to_soa_ext2(g_u64);
        let mut len = n;

        let use_parallel = n > EXT_PARALLEL_THRESHOLD;
        let mut sf_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sf_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };

        for round in 0..max_rounds {
            let (a_raw, b_raw) =
                crate::simd_sumcheck::reduce::ext2_soa_product_evaluate::<Backend>(
                    &f_c0[..len], &f_c1[..len], &g_c0[..len], &g_c1[..len], w,
                );
            let a: F = unsafe { ext_components_to_field(&a_raw) };
            let b: F = unsafe { ext_components_to_field(&b_raw) };
            let msg = (a, b);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);
            hook(round, transcript);
            let chg: F = transcript.read();
            verifier_messages.push(chg);

            let chg_raw: [u64; 2] = unsafe {
                let ptr = &chg as *const F as *const u64;
                [*ptr, *ptr.add(1)]
            };

            if len > EXT_PARALLEL_THRESHOLD {
                let new_len = len / 2;
                let _ = crate::simd_sumcheck::reduce::ext2_soa_product_reduce_and_evaluate_parallel::<Backend>(
                    &f_c0[..len], &f_c1[..len],
                    &g_c0[..len], &g_c1[..len],
                    &mut sf_c0[..new_len], &mut sf_c1[..new_len],
                    &mut sg_c0[..new_len], &mut sg_c1[..new_len],
                    chg_raw, w,
                );
                core::mem::swap(&mut f_c0, &mut sf_c0);
                core::mem::swap(&mut f_c1, &mut sf_c1);
                core::mem::swap(&mut g_c0, &mut sg_c0);
                core::mem::swap(&mut g_c1, &mut sg_c1);
                len = new_len;
            } else {
                let (_, _, new_len) =
                    crate::simd_sumcheck::reduce::ext2_soa_product_reduce_and_evaluate::<Backend>(
                        &mut f_c0[..len], &mut f_c1[..len],
                        &mut g_c0[..len], &mut g_c1[..len],
                        chg_raw, w,
                    );
                len = new_len;
            }
        }

        // SoA → AoS writeback into f and g, then truncate.
        let f_out: &mut [u64] = unsafe {
            core::slice::from_raw_parts_mut(f.as_mut_ptr() as *mut u64, len * d)
        };
        let g_out: &mut [u64] = unsafe {
            core::slice::from_raw_parts_mut(g.as_mut_ptr() as *mut u64, len * d)
        };
        for i in 0..len {
            f_out[2 * i] = f_c0[i];
            f_out[2 * i + 1] = f_c1[i];
            g_out[2 * i] = g_c0[i];
            g_out[2 * i + 1] = g_c1[i];
        }
        f.truncate(len);
        g.truncate(len);
    } else {
        // d == 3
        let w = extract_nonresidue_ext3::<F, Backend>();

        let (mut f_c0, mut f_c1, mut f_c2) = aos_to_soa_ext3(f_u64);
        let (mut g_c0, mut g_c1, mut g_c2) = aos_to_soa_ext3(g_u64);
        let mut len = n;

        let use_parallel = n > EXT_PARALLEL_THRESHOLD;
        let mut sf_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sf_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sf_c2: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c0: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c1: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };
        let mut sg_c2: Vec<u64> = if use_parallel { vec![0u64; n / 2] } else { Vec::new() };

        for round in 0..max_rounds {
            let (a_raw, b_raw) =
                crate::simd_sumcheck::reduce::ext3_soa_product_evaluate::<Backend>(
                    &f_c0[..len], &f_c1[..len], &f_c2[..len],
                    &g_c0[..len], &g_c1[..len], &g_c2[..len], w,
                );
            let a: F = unsafe { ext_components_to_field(&a_raw) };
            let b: F = unsafe { ext_components_to_field(&b_raw) };
            let msg = (a, b);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);
            hook(round, transcript);
            let chg: F = transcript.read();
            verifier_messages.push(chg);

            let chg_raw: [u64; 3] = unsafe {
                let ptr = &chg as *const F as *const u64;
                [*ptr, *ptr.add(1), *ptr.add(2)]
            };

            if len > EXT_PARALLEL_THRESHOLD {
                let new_len = len / 2;
                let _ = crate::simd_sumcheck::reduce::ext3_soa_product_reduce_and_evaluate_parallel::<Backend>(
                    &f_c0[..len], &f_c1[..len], &f_c2[..len],
                    &g_c0[..len], &g_c1[..len], &g_c2[..len],
                    &mut sf_c0[..new_len], &mut sf_c1[..new_len], &mut sf_c2[..new_len],
                    &mut sg_c0[..new_len], &mut sg_c1[..new_len], &mut sg_c2[..new_len],
                    chg_raw, w,
                );
                core::mem::swap(&mut f_c0, &mut sf_c0);
                core::mem::swap(&mut f_c1, &mut sf_c1);
                core::mem::swap(&mut f_c2, &mut sf_c2);
                core::mem::swap(&mut g_c0, &mut sg_c0);
                core::mem::swap(&mut g_c1, &mut sg_c1);
                core::mem::swap(&mut g_c2, &mut sg_c2);
                len = new_len;
            } else {
                let (_, _, new_len) =
                    crate::simd_sumcheck::reduce::ext3_soa_product_reduce_and_evaluate::<Backend>(
                        &mut f_c0[..len], &mut f_c1[..len], &mut f_c2[..len],
                        &mut g_c0[..len], &mut g_c1[..len], &mut g_c2[..len],
                        chg_raw, w,
                    );
                len = new_len;
            }
        }

        let f_out: &mut [u64] = unsafe {
            core::slice::from_raw_parts_mut(f.as_mut_ptr() as *mut u64, len * d)
        };
        let g_out: &mut [u64] = unsafe {
            core::slice::from_raw_parts_mut(g.as_mut_ptr() as *mut u64, len * d)
        };
        for i in 0..len {
            f_out[3 * i] = f_c0[i];
            f_out[3 * i + 1] = f_c1[i];
            f_out[3 * i + 2] = f_c2[i];
            g_out[3 * i] = g_c0[i];
            g_out[3 * i + 1] = g_c1[i];
            g_out[3 * i + 2] = g_c2[i];
        }
        f.truncate(len);
        g.truncate(len);
    }

    let final_evaluations = if f.len() == 1 {
        (f[0], g[0])
    } else {
        (F::ZERO, F::ZERO)
    };

    Some(crate::multilinear_product::ProductSumcheck {
        prover_messages,
        verifier_messages,
        final_evaluations,
    })
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
