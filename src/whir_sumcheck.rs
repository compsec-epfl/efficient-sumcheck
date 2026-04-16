//! WHIR-style quadratic inner-product sumcheck (faithful port).
//!
//! This is a straight port of the sumcheck prover/verifier used in
//! `compsec-epfl/whir` (see `whir/src/protocols/sumcheck.rs` and
//! `whir/src/algebra/sumcheck.rs`). The hot-loop algorithm is preserved
//! byte-for-byte; only the outer transcript interface is adapted to our
//! [`Transcript`](crate::transcript::Transcript) trait.
//!
//! Key differences vs [`crate::inner_product_sumcheck`]:
//!
//! - **Layout**: half-split. `a[0..n/2]` vs `a[n/2..]` is the split for the
//!   first variable (WHIR-native / MSB ordering). Callers do *not* need the
//!   MSB↔LSB bit-reversal reorder that our pair-split dispatch requires.
//! - **Transcript format**: `(c0, c2)` in difference form per round, with
//!   `c0 = q(0)` and `c2 = [x²] q(x)`. The verifier derives `c1` from the
//!   sumcheck constraint `q(0) + q(1) = sum`.
//! - **No SIMD dispatch**. Uses rayon `join` with a workload threshold —
//!   identical parallelism strategy to WHIR.
//! - **Staggered loop**: the round-`i` fold is deferred into round `i+1`
//!   and fused with that round's compute (via [`fold_and_compute_polynomial`]).
//!   The final challenge's fold happens once after the loop.
//!
//! Phase 1 of the WHIR-port plan: verify parity when dropped into `whir-effsc`.
//! Phase 2 will fuse `fold` + `compute` into a single pass (WHIR's own TODO),
//! and phase 3 will layer SIMD on top with a size threshold.

use ark_ff::Field;
#[cfg(feature = "parallel")]
use rayon::join;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::transcript::Transcript;

pub use crate::multilinear_product::ProductSumcheck;

// ─── Workload threshold ─────────────────────────────────────────────────────

/// Target single-thread workload size for `T`, mirroring `whir/src/utils.rs`.
/// Ideally a multiple of a cache line and close to L1 size.
const fn workload_size<T: Sized>() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const CACHE_SIZE: usize = 1 << 17; // 128 KiB Apple Silicon
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "ios", target_os = "android", target_os = "linux")
    ))]
    const CACHE_SIZE: usize = 1 << 16; // 64 KiB mobile/server ARM
    #[cfg(target_arch = "x86_64")]
    const CACHE_SIZE: usize = 1 << 15; // 32 KiB x86-64
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(
            target_arch = "aarch64",
            any(target_os = "ios", target_os = "android", target_os = "linux")
        ),
        target_arch = "x86_64"
    )))]
    const CACHE_SIZE: usize = 1 << 15;

    CACHE_SIZE / core::mem::size_of::<T>()
}

// ─── Scalar helpers ─────────────────────────────────────────────────────────

fn dot<F: Field>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "parallel")]
    if a.len() > workload_size::<F>() {
        return a.par_iter().zip(b).map(|(x, y)| *x * *y).sum();
    }
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

fn scalar_mul<F: Field>(v: &mut [F], w: F) {
    for x in v.iter_mut() {
        *x *= w;
    }
}

// ─── Core algebra (ported verbatim from whir/src/algebra/sumcheck.rs) ───────

/// Computes the constant and quadratic coefficient of the sumcheck polynomial.
///
/// Vectors `a` and `b` are implicitly zero-extended to the next power of two.
/// Returns `(c0, c2)` in difference form, where `q(x) = c0 + c1·x + c2·x²`.
pub fn compute_sumcheck_polynomial<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    fn recurse<F: Field>(a0: &[F], a1: &[F], b0: &[F], b1: &[F]) -> (F, F) {
        debug_assert_eq!(a0.len(), b0.len());
        debug_assert_eq!(a1.len(), b1.len());
        debug_assert!(a0.len() == a1.len());

        #[cfg(feature = "parallel")]
        if a0.len() * 4 > workload_size::<F>() {
            let mid = a0.len() / 2;
            let (a0l, a0r) = a0.split_at(mid);
            let (b0l, b0r) = b0.split_at(mid);
            let (a1l, a1r) = a1.split_at(mid);
            let (b1l, b1r) = b1.split_at(mid);
            let (left, right) = join(
                || recurse(a0l, a1l, b0l, b1l),
                || recurse(a0r, a1r, b0r, b1r),
            );
            return (left.0 + right.0, left.1 + right.1);
        }
        let mut acc0 = F::ZERO;
        let mut acc2 = F::ZERO;
        for ((&a0, &a1), (&b0, &b1)) in a0.iter().zip(a1).zip(b0.iter().zip(b1)) {
            acc0 += a0 * b0;
            acc2 += (a1 - a0) * (b1 - b0);
        }
        (acc0, acc2)
    }

    let non_padded = a.len().min(b.len());
    let a = &a[..non_padded];
    let b = &b[..non_padded];
    if a.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    if a.len() == 1 {
        return (a[0] * b[0], F::ZERO);
    }

    let half = a.len().next_power_of_two() >> 1;
    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half);
    debug_assert!(a0.len() >= a1.len());
    let (a0, a0_tail) = a0.split_at(a1.len());
    let (b0, b0_tail) = b0.split_at(a1.len());
    let (acc0, acc2) = recurse(a0, a1, b0, b1);

    // Tail part where a1, b1 is implicit zero padding. When a1 = b1 = 0,
    // both contributions collapse to a0·b0.
    let acc = dot(a0_tail, b0_tail);

    (acc0 + acc, acc2 + acc)
}

/// Folds evaluations by linear interpolation at `weight`, in place.
///
/// The `values` are implicitly zero-padded to the next power of two. On
/// return, the length is always a power of two (or zero).
pub fn fold<F: Field>(values: &mut Vec<F>, weight: F) {
    fn recurse_both<F: Field>(low: &mut [F], high: &[F], weight: F) {
        #[cfg(feature = "parallel")]
        if low.len() > workload_size::<F>() {
            let split = low.len() / 2;
            let (ll, lr) = low.split_at_mut(split);
            let (hl, hr) = high.split_at(split);
            join(
                || recurse_both(ll, hl, weight),
                || recurse_both(lr, hr, weight),
            );
            return;
        }
        for (low, high) in low.iter_mut().zip(high) {
            *low += (*high - *low) * weight;
        }
    }

    if values.len() <= 1 {
        return;
    }

    let half = values.len().next_power_of_two() >> 1;
    let (low, high) = values.split_at_mut(half);
    debug_assert!(low.len() >= high.len());
    let (low, tail) = low.split_at_mut(high.len());
    recurse_both(low, high, weight);

    // Tail where `high` is implicit zero padding: *low *= 1 - weight.
    scalar_mul(tail, F::ONE - weight);

    values.truncate(half);
    values.shrink_to_fit();
}

/// WHIR's two-pass fold-then-compute. Kept verbatim for the faithful port.
pub fn fold_and_compute_polynomial<F: Field>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    weight: F,
) -> (F, F) {
    fold(a, weight);
    fold(b, weight);
    compute_sumcheck_polynomial(a, b)
}

/// Single-pass fused variant. Folds `a` and `b` by `weight` *and* computes the
/// next-round polynomial `(c0, c2)` in one sweep over memory.
///
/// Layout observation: the fold splits at `L/2` and writes into `[0, L/2)`.
/// The subsequent compute splits the length-`L/2` folded vector at `L/4`. So
/// every quadruple `(a[k], a[k+L/4], a[k+L/2], a[k+3L/4])` is touched exactly
/// once — reading the old values, writing two folded values, and accumulating
/// the `(c0, c2)` contribution of the pair.
///
/// Memory traffic vs the unfused path: 8 reads + 4 writes per quadruple
/// (fused) instead of 12 reads + 4 writes (fold a + fold b + compute), a ~25%
/// reduction — most of the remaining headroom is from cache locality, since
/// all four strides are active simultaneously instead of in separate passes.
///
/// Falls back to the unfused path for small or non-pow2 inputs so the tail
/// accounting stays identical to WHIR's.
pub fn fused_fold_and_compute_polynomial<F: Field>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    weight: F,
) -> (F, F) {
    let l = a.len();
    debug_assert_eq!(l, b.len());
    if !l.is_power_of_two() || l < 4 {
        return fold_and_compute_polynomial(a, b, weight);
    }

    #[allow(clippy::too_many_arguments)]
    fn kernel<F: Field>(
        a0: &mut [F],
        a1: &mut [F],
        a2: &[F],
        a3: &[F],
        b0: &mut [F],
        b1: &mut [F],
        b2: &[F],
        b3: &[F],
        weight: F,
    ) -> (F, F) {
        debug_assert_eq!(a0.len(), a1.len());
        debug_assert_eq!(a0.len(), a2.len());
        debug_assert_eq!(a0.len(), a3.len());
        debug_assert_eq!(a0.len(), b0.len());
        debug_assert_eq!(a0.len(), b1.len());
        debug_assert_eq!(a0.len(), b2.len());
        debug_assert_eq!(a0.len(), b3.len());

        #[cfg(feature = "parallel")]
        if a0.len() * 4 > workload_size::<F>() {
            let mid = a0.len() / 2;
            let (a0l, a0r) = a0.split_at_mut(mid);
            let (a1l, a1r) = a1.split_at_mut(mid);
            let (a2l, a2r) = a2.split_at(mid);
            let (a3l, a3r) = a3.split_at(mid);
            let (b0l, b0r) = b0.split_at_mut(mid);
            let (b1l, b1r) = b1.split_at_mut(mid);
            let (b2l, b2r) = b2.split_at(mid);
            let (b3l, b3r) = b3.split_at(mid);
            let (left, right) = join(
                || kernel(a0l, a1l, a2l, a3l, b0l, b1l, b2l, b3l, weight),
                || kernel(a0r, a1r, a2r, a3r, b0r, b1r, b2r, b3r, weight),
            );
            return (left.0 + right.0, left.1 + right.1);
        }

        let mut c0 = F::ZERO;
        let mut c2 = F::ZERO;
        for i in 0..a0.len() {
            let x0 = a0[i];
            let x1 = a1[i];
            let x2 = a2[i];
            let x3 = a3[i];
            let y0 = b0[i];
            let y1 = b1[i];
            let y2 = b2[i];
            let y3 = b3[i];

            let na_lo = x0 + (x2 - x0) * weight;
            let na_hi = x1 + (x3 - x1) * weight;
            let nb_lo = y0 + (y2 - y0) * weight;
            let nb_hi = y1 + (y3 - y1) * weight;

            a0[i] = na_lo;
            a1[i] = na_hi;
            b0[i] = nb_lo;
            b1[i] = nb_hi;

            c0 += na_lo * nb_lo;
            c2 += (na_hi - na_lo) * (nb_hi - nb_lo);
        }
        (c0, c2)
    }

    let quarter = l / 4;
    let half = l / 2;

    let (a_first, a_second) = a.split_at_mut(half);
    let (a0, a1) = a_first.split_at_mut(quarter);
    let (a2, a3) = a_second.split_at(quarter);
    let (b_first, b_second) = b.split_at_mut(half);
    let (b0, b1) = b_first.split_at_mut(quarter);
    let (b2, b3) = b_second.split_at(quarter);

    let result = kernel(a0, a1, a2, a3, b0, b1, b2, b3, weight);

    a.truncate(half);
    b.truncate(half);
    // Note: unlike `fold`, we skip `shrink_to_fit` — the realloc/memcpy cost
    // is paid every round, whereas the capacity is freed once the Vec drops.
    result
}

// ─── Prover ─────────────────────────────────────────────────────────────────

/// Runs `num_rounds` rounds of WHIR's quadratic sumcheck on `(a, b)`, folding
/// both vectors in place.
///
/// Transcript format per round: writes `c0` then `c2` (difference form),
/// then invokes `hook(round, transcript)` (for per-round PoW grinding or
/// similar), then reads the verifier challenge.
///
/// Inputs follow WHIR's half-split layout — `a[0..n/2]` vs `a[n/2..]` is the
/// first-variable split. On return, if `num_rounds` reduces the input to
/// length 1, `final_evaluations = (a[0], b[0])`; otherwise `(F::ZERO, F::ZERO)`.
pub fn whir_sumcheck_partial_with_hook<F, T, H>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
    num_rounds: usize,
    hook: H,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    whir_sumcheck_partial_inner(a, b, transcript, num_rounds, hook, fold_and_compute_polynomial)
}

/// Same API as [`whir_sumcheck_partial_with_hook`] but uses the single-pass
/// [`fused_fold_and_compute_polynomial`] kernel. Semantically identical —
/// produces the same transcript bit-for-bit.
pub fn whir_sumcheck_fused_partial_with_hook<F, T, H>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
    num_rounds: usize,
    hook: H,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    whir_sumcheck_partial_inner(
        a,
        b,
        transcript,
        num_rounds,
        hook,
        fused_fold_and_compute_polynomial,
    )
}

fn whir_sumcheck_partial_inner<F, T, H, K>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
    num_rounds: usize,
    mut hook: H,
    mut fold_compute: K,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
    K: FnMut(&mut Vec<F>, &mut Vec<F>, F) -> (F, F),
{
    assert_eq!(a.len(), b.len());
    assert!(
        num_rounds == 0 || a.len().next_power_of_two() >= 1 << num_rounds,
        "num_rounds ({num_rounds}) exceeds log2 of next-pow2 of len ({})",
        a.len(),
    );

    let mut prover_messages: Vec<(F, F)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<F> = Vec::with_capacity(num_rounds);
    let mut folding_randomness: Option<F> = None;

    for round in 0..num_rounds {
        // Staggered: round-(i-1) fold is fused into round-i compute.
        let (c0, c2) = if let Some(w) = folding_randomness {
            fold_compute(a, b, w)
        } else {
            compute_sumcheck_polynomial(a, b)
        };

        prover_messages.push((c0, c2));
        transcript.write(c0);
        transcript.write(c2);

        hook(round, transcript);

        let r = transcript.read();
        verifier_messages.push(r);
        folding_randomness = Some(r);
    }

    if let Some(w) = folding_randomness {
        fold(a, w);
        fold(b, w);
    }

    let final_evaluations = if a.len() == 1 {
        (a[0], b[0])
    } else {
        (F::ZERO, F::ZERO)
    };

    ProductSumcheck {
        prover_messages,
        verifier_messages,
        final_evaluations,
    }
}

/// Convenience: runs a full sumcheck (`log2(next_pow2(len))` rounds) with a
/// per-round hook.
pub fn whir_sumcheck_with_hook<F, T, H>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
    hook: H,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    let num_rounds = if a.is_empty() {
        0
    } else {
        a.len().next_power_of_two().trailing_zeros() as usize
    };
    whir_sumcheck_partial_with_hook(a, b, transcript, num_rounds, hook)
}

/// Convenience: runs a full sumcheck with no per-round hook.
pub fn whir_sumcheck<F, T>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
{
    whir_sumcheck_with_hook(a, b, transcript, |_, _| {})
}

/// Fused variant of [`whir_sumcheck_with_hook`].
pub fn whir_sumcheck_fused_with_hook<F, T, H>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
    hook: H,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    let num_rounds = if a.is_empty() {
        0
    } else {
        a.len().next_power_of_two().trailing_zeros() as usize
    };
    whir_sumcheck_fused_partial_with_hook(a, b, transcript, num_rounds, hook)
}

/// Fused variant of [`whir_sumcheck`].
pub fn whir_sumcheck_fused<F, T>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
{
    whir_sumcheck_fused_with_hook(a, b, transcript, |_, _| {})
}

// ─── Verifier ───────────────────────────────────────────────────────────────

/// Runs the verifier side of [`whir_sumcheck_partial_with_hook`]. Reads
/// `(c0, c2)` per round, derives `c1 = sum - 2·c0 - c2`, calls
/// `hook(round, transcript)` (for per-round PoW verification), reads the
/// challenge, and updates `sum` by Horner evaluation `(c2·r + c1)·r + c0`.
///
/// Returns the sampled challenges. On return, `*sum` is the claim reduced
/// to the final folded point.
pub fn whir_sumcheck_verify_with_hook<F, T, H>(
    transcript: &mut T,
    sum: &mut F,
    num_rounds: usize,
    mut hook: H,
) -> Vec<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    let mut res = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let c0: F = transcript.read();
        let c2: F = transcript.read();
        let c1 = *sum - c0.double() - c2;

        hook(round, transcript);

        let r = transcript.read();
        res.push(r);
        *sum = (c2 * r + c1) * r + c0;
    }
    res
}

/// Convenience wrapper over [`whir_sumcheck_verify_with_hook`] with no hook.
pub fn whir_sumcheck_verify<F, T>(
    transcript: &mut T,
    sum: &mut F,
    num_rounds: usize,
) -> Vec<F>
where
    F: Field,
    T: Transcript<F>,
{
    whir_sumcheck_verify_with_hook(transcript, sum, num_rounds, |_, _| {})
}

// Tests live in `tests/whir_sumcheck.rs` (integration target) because the
// sibling test modules currently fail to compile against the pinned
// spongefish revision, which blocks the whole lib-test target.
