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

    if n <= HYBRID_THRESHOLD {
        dispatch_all_simd::<BF, EF, Backend>(
            evaluations,
            transcript,
            num_rounds,
            &mut prover_messages,
            &mut verifier_messages,
        );
    } else {
        dispatch_hybrid::<BF, EF, Backend>(
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
/// Best for small-to-medium inputs where SIMD reduce beats generic.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
fn dispatch_all_simd<
    BF: Field,
    EF: Field + From<BF>,
    S: crate::simd_fields::SimdBaseField<Scalar = u64>,
>(
    evaluations: &mut [BF],
    transcript: &mut impl Transcript<EF>,
    num_rounds: usize,
    prover_messages: &mut Vec<(EF, EF)>,
    verifier_messages: &mut Vec<EF>,
) {
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
        }
    }
}

/// Hybrid path: SIMD evaluate + generic arkworks reduce.
/// Best for large inputs where rayon-parallel Field reduce dominates.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
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

// ─── Inner product dispatch ─────────────────────────────────────────────────

/// Try to run the inner product sumcheck on the SIMD backend.
///
/// Same safety invariant as [`try_simd_dispatch`].
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
pub(crate) fn try_simd_product_dispatch<BF: Field, EF: Field + From<BF>>(
    f: &mut [BF],
    g: &mut [BF],
    transcript: &mut impl Transcript<EF>,
) -> Option<crate::multilinear_product::ProductSumcheck<EF>> {
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
    use crate::simd_sumcheck::reduce::reduce_in_place;

    let n = f.len();
    let num_rounds = n.trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<EF> = Vec::with_capacity(num_rounds);

    if num_rounds > 0 {
        // BF == EF (both Goldilocks): work in-place on the original buffers.
        // No cross_field_reduce allocation needed.
        let f_raw: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(f.as_mut_ptr() as *mut u64, n) };
        let g_raw: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(g.as_mut_ptr() as *mut u64, n) };

        let mut f_len = n;
        let mut g_len = n;

        for round in 0..num_rounds {
            let (a, b) = product_evaluate_parallel::<Backend>(&f_raw[..f_len], &g_raw[..g_len]);

            let msg = (u64_to_field::<EF>(a), u64_to_field::<EF>(b));
            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            let chg_ef: EF = transcript.read();
            verifier_messages.push(chg_ef);

            if round < num_rounds - 1 {
                let chg: u64 = field_to_u64(chg_ef);
                f_len = reduce_in_place::<Backend>(&mut f_raw[..f_len], chg);
                g_len = reduce_in_place::<Backend>(&mut g_raw[..g_len], chg);
            }
        }
    }

    Some(crate::multilinear_product::ProductSumcheck {
        verifier_messages,
        prover_messages,
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

        let n_u64 = evals.len() * d;
        let buf: &[u64] =
            unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, n_u64) };

        // Extract challenge components as raw u64
        let chg_raw: [u64; 2] = unsafe {
            let ptr = &challenge as *const EF as *const u64;
            [*ptr, *ptr.add(1)]
        };

        // Extract nonresidue from the extension field config.
        // We compute (0, 1) * (0, 1) = (NONRESIDUE, 0) to get it at runtime.
        let one_x = unsafe {
            use crate::simd_fields::SimdBaseField;
            let mut tmp = [0u64; 2];
            tmp[1] = Backend::ONE; // c1 = 1 (in Montgomery form)
            let one_x: EF = core::mem::transmute_copy(&tmp);
            one_x
        };
        let nr = one_x * one_x;
        let w: u64 = unsafe { *((&nr) as *const EF as *const u64) };

        let result_u64 = crate::simd_sumcheck::reduce::ext2_reduce_parallel(buf, chg_raw, w);

        // Reinterpret result u64s as EF elements
        let new_len = result_u64.len() / d;
        let result_ef: Vec<EF> = unsafe {
            let mut v = core::mem::ManuallyDrop::new(result_u64);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut EF, new_len, v.capacity() / d)
        };
        *evals = result_ef;
        return true;
    }

    // degree 3, 4, etc. — fall through to generic
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
