//! Quadratic inner-product sumcheck: `∑_x f(x)·g(x)`.
//!
//! Half-split (MSB) layout with a fused fold+compute kernel.
//! Round `i` folds the top-most remaining variable — the split is over
//! `a[0..L/2]` vs `a[L/2..L]`, *not* the adjacent pairs `(a[2k], a[2k+1])`
//! of a pair-split (LSB) layout. Callers whose upstream indexing assumed
//! pair-split semantics must reorder their inputs with a bit-reversal.
//!
//! Wire format per round: `(c0, c2)` in *difference form*, where
//!   - `c0 = q(0) = Σ a_lo·b_lo`
//!   - `c2 = [x²] q(x) = Σ (a_hi − a_lo)·(b_hi − b_lo)`
//!
//! The verifier derives `c1 = claim − 2·c0 − c2` from the sumcheck
//! constraint `q(0) + q(1) = claim`.
//!
//! The fused kernel rolls the round-`i` fold into the round-`(i+1)` compute,
//! cutting memory traffic from 12 reads + 4 writes per quadruple to
//! 8 reads + 4 writes — roughly a 25% reduction on the cold path, with
//! additional cache-locality gains from reading all four strides
//! simultaneously.

use ark_ff::Field;
#[cfg(feature = "parallel")]
use rayon::join;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::transcript::Transcript;

pub use crate::multilinear_product::ProductSumcheck;

// ─── Workload threshold ─────────────────────────────────────────────────────

/// Target single-thread workload size for `T`. Close to L1 cache.
const fn workload_size<T: Sized>() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const CACHE_SIZE: usize = 1 << 17;
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "ios", target_os = "android", target_os = "linux")
    ))]
    const CACHE_SIZE: usize = 1 << 16;
    #[cfg(target_arch = "x86_64")]
    const CACHE_SIZE: usize = 1 << 15;
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

// ─── Core algebra ───────────────────────────────────────────────────────────

/// `(c0, c2)` of the round polynomial `q(x) = c0 + c1·x + c2·x²`.
///
/// Vectors `a` and `b` are implicitly zero-extended to the next power of two.
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

    // Tail (a1, b1 = implicit zero padding): both contributions collapse to a0·b0.
    let acc = dot(a0_tail, b0_tail);
    (acc0 + acc, acc2 + acc)
}

/// In-place half-split fold: `new[k] = v[k] + (v[k+L/2] − v[k]) · weight`.
///
/// `values` is implicitly zero-padded to the next power of two. On return,
/// the length is a power of two (or zero).
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

    // Tail with implicit zero high: *low *= 1 − weight.
    scalar_mul(tail, F::ONE - weight);

    values.truncate(half);
    values.shrink_to_fit();
}

/// Two-pass fold-then-compute; reference version kept for testing.
pub fn fold_and_compute_polynomial<F: Field>(a: &mut Vec<F>, b: &mut Vec<F>, weight: F) -> (F, F) {
    fold(a, weight);
    fold(b, weight);
    compute_sumcheck_polynomial(a, b)
}

/// Fused single-pass variant.
///
/// Folds `a` and `b` by `weight` *and* computes the next-round polynomial
/// `(c0, c2)` in one sweep. The fold writes `[0, L/2)`; the subsequent
/// compute splits the length-`L/2` folded vector at `L/4`. So every
/// quadruple `(x[k], x[k+L/4], x[k+L/2], x[k+3L/4])` is touched exactly
/// once — 8 reads + 4 writes (fused) vs. 12 reads + 4 writes (unfused).
///
/// Falls back to the unfused path for small or non-pow2 inputs so the
/// implicit-zero tail accounting stays identical.
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
    // Skip shrink_to_fit — realloc per round is pricier than the capacity
    // we carry; the capacity frees once the Vec drops.
    result
}

// ─── Prover ─────────────────────────────────────────────────────────────────

/// Runs `num_rounds` rounds on `(a, b)`, folding both in place.
///
/// Transcript per round: writes `c0` then `c2` (difference form), invokes
/// `hook(round, transcript)`, then reads the verifier challenge.
///
/// On return, if `num_rounds == log2(next_pow2(len))` then `a` and `b` have
/// length 1 and `final_evaluations = (a[0], b[0])`; otherwise
/// `(F::ZERO, F::ZERO)`.
pub fn inner_product_sumcheck_partial_with_hook<F, T, H>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
    num_rounds: usize,
    mut hook: H,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
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
            fused_fold_and_compute_polynomial(a, b, w)
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

/// Full sumcheck (`log2(next_pow2(len))` rounds) with a per-round hook.
pub fn inner_product_sumcheck_with_hook<F, T, H>(
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
    inner_product_sumcheck_partial_with_hook(a, b, transcript, num_rounds, hook)
}

/// Full sumcheck with no per-round hook.
pub fn inner_product_sumcheck<F, T>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    transcript: &mut T,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
{
    inner_product_sumcheck_with_hook(a, b, transcript, |_, _| {})
}

// ─── Verifier ───────────────────────────────────────────────────────────────

/// Verifier side of [`inner_product_sumcheck_with_hook`].
///
/// Reads `(c0, c2)` per round, derives `c1 = sum − 2·c0 − c2`, calls
/// `hook(round, transcript)`, reads the challenge, and updates `sum` by
/// Horner evaluation `(c2·r + c1)·r + c0`. Returns the sampled challenges;
/// `*sum` is the claim reduced to the final folded point.
pub fn inner_product_sumcheck_verify_with_hook<F, T, H>(
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

/// Convenience wrapper over [`inner_product_sumcheck_verify_with_hook`] with no hook.
pub fn inner_product_sumcheck_verify<F, T>(
    transcript: &mut T,
    sum: &mut F,
    num_rounds: usize,
) -> Vec<F>
where
    F: Field,
    T: Transcript<F>,
{
    inner_product_sumcheck_verify_with_hook(transcript, sum, num_rounds, |_, _| {})
}

// Tests live in `tests/inner_product_sumcheck.rs` (integration target) —
// the lib-test target is blocked by unrelated modules with stale
// `domain_separator!` syntax.
