//! Standard multilinear sumcheck: `∑_x v(x)`.
//!
//! Half-split (MSB) layout with a fused fold+compute kernel. Round `i`
//! folds the top-most remaining variable — the round-0 split is
//! `v[0..L/2]` vs `v[L/2..L]`, *not* the adjacent pairs `(v[2k], v[2k+1])`
//! of a pair-split (LSB) layout. Callers whose upstream indexing assumed
//! pair-split semantics must reorder their inputs with a bit-reversal.
//!
//! Wire format per round: `(s0, s1)` where
//!   - `s0 = q(0) = Σ v_lo`
//!   - `s1 = q(1) = Σ v_hi`
//!
//! The round polynomial is degree 1: `q(X) = s0 + X·(s1 − s0)`. Consistency
//! invariant: `s0 + s1 == current_claim`.
//!
//! The fused kernel rolls the round-`i` fold into the round-`(i+1)` compute:
//! 4 reads + 2 writes per quadruple (fused) vs. 6 reads + 2 writes
//! (fold + compute separately) — a ~33% memory-traffic reduction.

use ark_ff::Field;
#[cfg(feature = "parallel")]
use rayon::join;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::transcript::{ProverTranscript, VerifierTranscript};

/// Legacy return type for `multilinear_sumcheck`.
#[derive(Debug)]
pub struct Sumcheck<F: Field> {
    pub prover_messages: Vec<(F, F)>,
    pub verifier_messages: Vec<F>,
    pub final_evaluation: F,
}

// ─── Workload threshold ─────────────────────────────────────────────────────

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

fn sum_slice<F: Field>(v: &[F]) -> F {
    #[cfg(feature = "parallel")]
    if v.len() > workload_size::<F>() {
        return v.par_iter().copied().sum();
    }
    v.iter().copied().sum()
}

fn scalar_mul<F: Field>(v: &mut [F], w: F) {
    for x in v.iter_mut() {
        *x *= w;
    }
}

// ─── Core algebra ───────────────────────────────────────────────────────────

/// `(s0, s1)` of the degree-1 round polynomial `q(X) = s0 + X·(s1 − s0)`.
///
/// `values` is implicitly zero-extended to the next power of two.
///   - `s0 = Σ v[0..L/2]` (low half, possibly with tail contributions)
///   - `s1 = Σ v[L/2..L]`
pub fn compute_sumcheck_polynomial<F: Field>(values: &[F]) -> (F, F) {
    fn recurse<F: Field>(lo: &[F], hi: &[F]) -> (F, F) {
        debug_assert_eq!(lo.len(), hi.len());

        #[cfg(feature = "parallel")]
        if lo.len() * 2 > workload_size::<F>() {
            let mid = lo.len() / 2;
            let (lol, lor) = lo.split_at(mid);
            let (hil, hir) = hi.split_at(mid);
            let (l, r) = join(|| recurse(lol, hil), || recurse(lor, hir));
            return (l.0 + r.0, l.1 + r.1);
        }
        let mut s0 = F::ZERO;
        let mut s1 = F::ZERO;
        for (&l, &h) in lo.iter().zip(hi) {
            s0 += l;
            s1 += h;
        }
        (s0, s1)
    }

    if values.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    if values.len() == 1 {
        // Implicit zero pad on the high half: (v[0], 0).
        return (values[0], F::ZERO);
    }

    let half = values.len().next_power_of_two() >> 1;
    let (lo, hi) = values.split_at(half);
    debug_assert!(lo.len() >= hi.len());
    let (lo, lo_tail) = lo.split_at(hi.len());
    let (s0, s1) = recurse(lo, hi);

    // Tail (hi implicitly zero): contributes to s0 only.
    let tail = sum_slice(lo_tail);
    (s0 + tail, s1)
}

/// In-place half-split (MSB) fold: `new[k] = v[k] + (v[k+L/2] − v[k]) · weight`.
///
/// Implicit zero padding on the high half collapses the tail to `v[k] * (1 − w)`.
///
/// SIMD-accelerated for Goldilocks base field on NEON and AVX-512 IFMA.
/// Falls back to a scalar recursive `rayon::join` fold for other fields.
pub fn fold<F: Field>(values: &mut Vec<F>, weight: F) {
    // SIMD fast path for base-field Goldilocks (MSB layout).
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx512ifma")
        )
    ))]
    {
        if crate::simd_sumcheck::dispatch::try_simd_reduce_msb(values, weight) {
            values.shrink_to_fit();
            return;
        }
    }
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

    scalar_mul(tail, F::ONE - weight);

    values.truncate(half);
    values.shrink_to_fit();
}

/// Two-pass fold-then-compute. Reference only.
pub fn fold_and_compute_polynomial<F: Field>(values: &mut Vec<F>, weight: F) -> (F, F) {
    fold(values, weight);
    compute_sumcheck_polynomial(values)
}

/// Fused fold + compute: folds `values` by `weight` *and* returns the
/// next-round `(s0, s1)` in one sweep over the quadruple
/// `(v[k], v[k+L/4], v[k+L/2], v[k+3L/4])`.
pub fn fused_fold_and_compute_polynomial<F: Field>(values: &mut Vec<F>, weight: F) -> (F, F) {
    let l = values.len();
    if !l.is_power_of_two() || l < 4 {
        return fold_and_compute_polynomial(values, weight);
    }

    fn kernel<F: Field>(v0: &mut [F], v1: &mut [F], v2: &[F], v3: &[F], weight: F) -> (F, F) {
        debug_assert_eq!(v0.len(), v1.len());
        debug_assert_eq!(v0.len(), v2.len());
        debug_assert_eq!(v0.len(), v3.len());

        #[cfg(feature = "parallel")]
        if v0.len() * 2 > workload_size::<F>() {
            let mid = v0.len() / 2;
            let (v0l, v0r) = v0.split_at_mut(mid);
            let (v1l, v1r) = v1.split_at_mut(mid);
            let (v2l, v2r) = v2.split_at(mid);
            let (v3l, v3r) = v3.split_at(mid);
            let (left, right) = join(
                || kernel(v0l, v1l, v2l, v3l, weight),
                || kernel(v0r, v1r, v2r, v3r, weight),
            );
            return (left.0 + right.0, left.1 + right.1);
        }

        let mut s0 = F::ZERO;
        let mut s1 = F::ZERO;
        for i in 0..v0.len() {
            let x0 = v0[i];
            let x1 = v1[i];
            let x2 = v2[i];
            let x3 = v3[i];

            let n_lo = x0 + (x2 - x0) * weight;
            let n_hi = x1 + (x3 - x1) * weight;

            v0[i] = n_lo;
            v1[i] = n_hi;

            s0 += n_lo;
            s1 += n_hi;
        }
        (s0, s1)
    }

    let quarter = l / 4;
    let half = l / 2;

    let (first, second) = values.split_at_mut(half);
    let (v0, v1) = first.split_at_mut(quarter);
    let (v2, v3) = second.split_at(quarter);

    let result = kernel(v0, v1, v2, v3, weight);

    values.truncate(half);
    result
}

// ─── Prover ─────────────────────────────────────────────────────────────────

/// Runs `num_rounds` rounds on `values`, folding it in place.
///
/// Transcript per round: writes `s0` then `s1`, invokes
/// `hook(round, transcript)`, then reads the verifier challenge.
///
/// On return, if `num_rounds == log2(next_pow2(len))` then `values.len() == 1`
/// and `final_evaluation = values[0]`; otherwise `F::ZERO`.
pub fn multilinear_sumcheck_partial<F, T, H>(
    values: &mut Vec<F>,
    transcript: &mut T,
    num_rounds: usize,
    mut hook: H,
) -> Sumcheck<F>
where
    F: Field,
    T: ProverTranscript<F>,
    H: FnMut(usize, &mut T),
{
    assert!(
        num_rounds == 0 || values.len().next_power_of_two() >= 1 << num_rounds,
        "num_rounds ({num_rounds}) exceeds log2 of next-pow2 of len ({})",
        values.len(),
    );

    let mut prover_messages: Vec<(F, F)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<F> = Vec::with_capacity(num_rounds);
    let mut folding_randomness: Option<F> = None;

    for round in 0..num_rounds {
        let (s0, s1) = if let Some(w) = folding_randomness {
            fused_fold_and_compute_polynomial(values, w)
        } else {
            compute_sumcheck_polynomial(values)
        };

        prover_messages.push((s0, s1));
        transcript.send(s0);
        transcript.send(s1);

        hook(round, transcript);

        let r = transcript.challenge();
        verifier_messages.push(r);
        folding_randomness = Some(r);
    }

    if let Some(w) = folding_randomness {
        fold(values, w);
    }

    let final_evaluation = if values.len() == 1 {
        values[0]
    } else {
        F::ZERO
    };

    Sumcheck {
        prover_messages,
        verifier_messages,
        final_evaluation,
    }
}

/// Full sumcheck (`log2(next_pow2(len))` rounds) with a per-round hook.
pub fn multilinear_sumcheck<F, T, H>(
    values: &mut Vec<F>,
    transcript: &mut T,
    hook: H,
) -> Sumcheck<F>
where
    F: Field,
    T: ProverTranscript<F>,
    H: FnMut(usize, &mut T),
{
    let num_rounds = if values.is_empty() {
        0
    } else {
        values.len().next_power_of_two().trailing_zeros() as usize
    };
    multilinear_sumcheck_partial(values, transcript, num_rounds, hook)
}

// ─── Verifier ───────────────────────────────────────────────────────────────

/// Verifier side. Reads `(s0, s1)` per round, checks `s0 + s1 == *sum`,
/// invokes `hook(round, transcript)?`, reads the challenge, and updates
/// `*sum = s0 + r·(s1 − s0)`. Returns the sampled challenges on success.
pub fn multilinear_sumcheck_verify<F, T, H>(
    transcript: &mut T,
    sum: &mut F,
    num_rounds: usize,
    mut hook: H,
) -> Result<Vec<F>, crate::proof::SumcheckError>
where
    F: Field,
    T: VerifierTranscript<F>,
    H: FnMut(usize, &mut T) -> Result<(), crate::proof::SumcheckError>,
{
    let mut res = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let s0: F = transcript
            .receive()
            .map_err(|_| crate::proof::SumcheckError::TranscriptError { round })?;
        let s1: F = transcript
            .receive()
            .map_err(|_| crate::proof::SumcheckError::TranscriptError { round })?;
        if s0 + s1 != *sum {
            return Err(crate::proof::SumcheckError::ConsistencyCheck { round });
        }

        hook(round, transcript)?;

        let r = transcript.challenge();
        res.push(r);
        *sum = s0 + r * (s1 - s0);
    }
    Ok(res)
}

// Tests live in `tests/multilinear_sumcheck.rs` (integration target).
