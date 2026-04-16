//! Integration tests for the ported WHIR sumcheck.
//!
//! Kept out of the library's inline `#[cfg(test)]` blocks because the
//! sibling test modules (inner_product_sumcheck, multilinear_sumcheck,
//! coefficient_sumcheck) currently fail to compile against the pinned
//! spongefish revision (stale `domain_separator!` syntax), which blocks
//! the whole lib-test target. Integration tests only need the `lib`
//! target to build, so they're unaffected.

use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use efficient_sumcheck::tests::F64;
use efficient_sumcheck::transcript::SanityTranscript;
use efficient_sumcheck::{
    whir_sumcheck, whir_sumcheck_fused, whir_sumcheck_partial_with_hook, whir_sumcheck_with_hook,
};

const SEED: u64 = 0xA110C8ED;

fn rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

fn dot_ref<F: Field>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

/// Evaluate the multilinear extension of `evals` at `point`, following
/// WHIR's half-split / MSB ordering: each round pops the top half of the
/// vector and linearly interpolates against the bottom half.
fn multilinear_extend<F: Field>(evals: &[F], point: &[F]) -> F {
    assert_eq!(evals.len(), 1 << point.len());
    let mut current = evals.to_vec();
    for &r in point {
        let half = current.len() / 2;
        let (low, high) = current.split_at(half);
        current = low
            .iter()
            .zip(high)
            .map(|(l, h)| *l + (*h - *l) * r)
            .collect();
    }
    current[0]
}

#[test]
fn test_power_of_two_roundtrip() {
    let num_vars = 8;
    let n = 1 << num_vars;

    let mut r = rng();
    let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let initial_sum = dot_ref(&a_orig, &b_orig);

    // Prover — SanityTranscript ignores writes and reads random challenges
    // from a seeded RNG, so a fresh SanityTranscript with the same seed
    // reproduces the exact challenge sequence on the verifier side.
    let mut prover_rng = rng();
    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut t_prove = SanityTranscript::new(&mut prover_rng);
    let result = whir_sumcheck(&mut a, &mut b, &mut t_prove);

    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(result.prover_messages.len(), num_vars);
    assert_eq!(result.verifier_messages.len(), num_vars);
    assert_eq!(result.final_evaluations, (a[0], b[0]));

    // SanityTranscript discards writes and draws reads from its RNG, so it
    // can't round-trip a real Fiat-Shamir verifier. We check prover-side
    // consistency instead: the folded values `(a[0], b[0])` match an
    // independent multilinear extension of the originals at the verifier
    // challenges produced by the prover run.
    let _ = initial_sum;
    assert_eq!(multilinear_extend(&a_orig, &result.verifier_messages), a[0]);
    assert_eq!(multilinear_extend(&b_orig, &result.verifier_messages), b[0]);
}

#[test]
fn test_non_power_of_two_partial_runs() {
    // We can't cleanly round-trip verify through SanityTranscript, but we
    // can confirm the prover runs to completion over non-pow2 inputs with
    // the WHIR padding semantics and produces the expected message count.
    let initial_size = 13_usize;
    let padded = initial_size.next_power_of_two();
    let num_rounds = padded.trailing_zeros() as usize;

    let mut r = rng();
    let a_orig: Vec<F64> = (0..initial_size).map(|_| F64::rand(&mut r)).collect();
    let b_orig: Vec<F64> = (0..initial_size).map(|_| F64::rand(&mut r)).collect();

    let mut prover_rng = rng();
    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut t = SanityTranscript::new(&mut prover_rng);
    let result =
        whir_sumcheck_partial_with_hook(&mut a, &mut b, &mut t, num_rounds, |_, _| {});
    assert_eq!(result.prover_messages.len(), num_rounds);
    assert_eq!(result.verifier_messages.len(), num_rounds);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
}

#[test]
fn test_partial_split_matches_full() {
    // partial(k) then partial(n − k) produces the same transcript as one
    // full run, and the second partial's `final_evaluations` equals the
    // full run's.
    let num_vars = 8;
    let n = 1 << num_vars;
    let split_at = 3;

    let mut r = rng();
    let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut a_full = a_orig.clone();
    let mut b_full = b_orig.clone();
    let mut full_rng = rng();
    let mut t_full = SanityTranscript::new(&mut full_rng);
    let full = whir_sumcheck(&mut a_full, &mut b_full, &mut t_full);

    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut split_rng = rng();
    let mut t_split = SanityTranscript::new(&mut split_rng);
    let first =
        whir_sumcheck_partial_with_hook(&mut a, &mut b, &mut t_split, split_at, |_, _| {});
    let second = whir_sumcheck_partial_with_hook(
        &mut a,
        &mut b,
        &mut t_split,
        num_vars - split_at,
        |_, _| {},
    );

    let mut split_prover = first.prover_messages.clone();
    split_prover.extend(second.prover_messages.iter().copied());
    let mut split_verifier = first.verifier_messages.clone();
    split_verifier.extend(second.verifier_messages.iter().copied());

    assert_eq!(split_prover, full.prover_messages);
    assert_eq!(split_verifier, full.verifier_messages);
    assert_eq!(second.final_evaluations, full.final_evaluations);
    assert_eq!(first.final_evaluations, (F64::ZERO, F64::ZERO));
}

#[test]
fn test_hook_called_once_per_round() {
    use std::cell::RefCell;
    let num_vars = 6;
    let n = 1 << num_vars;

    let mut r = rng();
    let mut a: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let mut b: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);

    let calls = RefCell::new(Vec::<usize>::new());
    let result = whir_sumcheck_with_hook(&mut a, &mut b, &mut t, |round, _| {
        calls.borrow_mut().push(round);
    });
    assert_eq!(result.prover_messages.len(), num_vars);
    assert_eq!(calls.into_inner(), (0..num_vars).collect::<Vec<_>>());
}

#[test]
fn test_zero_rounds_is_identity() {
    let mut r = rng();
    let a_orig: Vec<F64> = (0..8).map(|_| F64::rand(&mut r)).collect();
    let b_orig: Vec<F64> = (0..8).map(|_| F64::rand(&mut r)).collect();
    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);

    let result = whir_sumcheck_partial_with_hook(&mut a, &mut b, &mut t, 0, |_, _| {});
    assert!(result.prover_messages.is_empty());
    assert!(result.verifier_messages.is_empty());
    assert_eq!(a, a_orig);
    assert_eq!(b, b_orig);
}

#[test]
fn test_prover_msg_is_difference_form() {
    // Round-0 message (c0, c2) must be in difference form:
    //   c0 = Σ a_lo · b_lo                    (= q(0))
    //   c2 = Σ (a_hi − a_lo)·(b_hi − b_lo)    (= [x²] q(x))
    // so the verifier's `c1 = sum − 2·c0 − c2` derivation is correct.
    let n = 16_usize;
    let mut r = rng();
    let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut a_mut = a.clone();
    let mut b_mut = b.clone();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let result =
        whir_sumcheck_partial_with_hook(&mut a_mut, &mut b_mut, &mut t, 1, |_, _| {});
    let (c0, c2) = result.prover_messages[0];

    let half = n / 2;
    let expected_c0: F64 = a[..half].iter().zip(&b[..half]).map(|(x, y)| *x * *y).sum();
    let expected_c2: F64 = a[..half]
        .iter()
        .zip(&a[half..])
        .zip(b[..half].iter().zip(&b[half..]))
        .map(|((a0, a1), (b0, b1))| (*a1 - *a0) * (*b1 - *b0))
        .sum();
    assert_eq!(c0, expected_c0);
    assert_eq!(c2, expected_c2);
}

#[test]
fn test_deterministic_under_same_seed() {
    // Two independent runs with the same seed produce identical transcripts.
    let n = 1 << 5;
    let mut r = rng();
    let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let run = || -> _ {
        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        let mut trng = rng();
        let mut t = SanityTranscript::new(&mut trng);
        whir_sumcheck(&mut a, &mut b, &mut t)
    };
    let r1 = run();
    let r2 = run();
    assert_eq!(r1.prover_messages, r2.prover_messages);
    assert_eq!(r1.verifier_messages, r2.verifier_messages);
    assert_eq!(r1.final_evaluations, r2.final_evaluations);
}

#[test]
fn test_fused_matches_faithful_pow2() {
    // The fused kernel must produce bit-identical transcripts and folds to
    // the faithful (unfused) path for pow2 inputs — otherwise the fusion
    // arithmetic has drifted.
    for &num_vars in &[1_usize, 2, 4, 7, 10] {
        let n = 1 << num_vars;
        let mut r = rng();
        let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
        let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

        let mut a1 = a_orig.clone();
        let mut b1 = b_orig.clone();
        let mut rng1 = rng();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let faithful = whir_sumcheck(&mut a1, &mut b1, &mut t1);

        let mut a2 = a_orig.clone();
        let mut b2 = b_orig.clone();
        let mut rng2 = rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let fused = whir_sumcheck_fused(&mut a2, &mut b2, &mut t2);

        assert_eq!(faithful.prover_messages, fused.prover_messages, "n={n}");
        assert_eq!(faithful.verifier_messages, fused.verifier_messages, "n={n}");
        assert_eq!(faithful.final_evaluations, fused.final_evaluations, "n={n}");
        assert_eq!(a1, a2, "folded a mismatch at n={n}");
        assert_eq!(b1, b2, "folded b mismatch at n={n}");
    }
}

#[test]
fn test_fused_matches_faithful_non_pow2() {
    // Non-pow2 inputs fall back to the unfused path inside the fused kernel;
    // verify the fallback is transparent.
    for &n in &[3_usize, 5, 13, 33, 100] {
        let mut r = rng();
        let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
        let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

        let mut a1 = a_orig.clone();
        let mut b1 = b_orig.clone();
        let mut rng1 = rng();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let faithful = whir_sumcheck(&mut a1, &mut b1, &mut t1);

        let mut a2 = a_orig.clone();
        let mut b2 = b_orig.clone();
        let mut rng2 = rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let fused = whir_sumcheck_fused(&mut a2, &mut b2, &mut t2);

        assert_eq!(faithful.prover_messages, fused.prover_messages, "n={n}");
        assert_eq!(faithful.verifier_messages, fused.verifier_messages, "n={n}");
        assert_eq!(faithful.final_evaluations, fused.final_evaluations, "n={n}");
    }
}

// Silence unused-import warning when this crate is built without tests
// exercising AdditiveGroup. (Referenced in F64::ZERO below.)
const _: F64 = <F64 as AdditiveGroup>::ZERO;
