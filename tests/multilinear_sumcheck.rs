//! Integration tests for the MSB fused multilinear sumcheck.

use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use effsc::tests::F64;
use effsc::transcript::{SanityTranscript, Transcript};
use effsc::{multilinear_sumcheck, multilinear_sumcheck_partial, Sumcheck};

const SEED: u64 = 0xA110C8ED;

fn rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

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
    let v_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut prover_rng = rng();
    let mut v = v_orig.clone();
    let mut t_prove = SanityTranscript::new(&mut prover_rng);
    let result: Sumcheck<F64> = multilinear_sumcheck(&mut v, &mut t_prove, |_, _| {});

    assert_eq!(v.len(), 1);
    assert_eq!(result.prover_messages.len(), num_vars);
    assert_eq!(result.verifier_messages.len(), num_vars);
    assert_eq!(result.final_evaluation, v[0]);

    // Folded value matches an independent MLE evaluation.
    assert_eq!(multilinear_extend(&v_orig, &result.verifier_messages), v[0]);

    // Round-0 consistency: s0 + s1 == Σ v.
    let claim: F64 = v_orig.iter().copied().sum();
    let (s0, s1) = result.prover_messages[0];
    assert_eq!(s0 + s1, claim);
}

#[test]
fn test_non_power_of_two_partial_runs() {
    let initial_size = 13_usize;
    let padded = initial_size.next_power_of_two();
    let num_rounds = padded.trailing_zeros() as usize;

    let mut r = rng();
    let v_orig: Vec<F64> = (0..initial_size).map(|_| F64::rand(&mut r)).collect();

    let mut prover_rng = rng();
    let mut v = v_orig.clone();
    let mut t = SanityTranscript::new(&mut prover_rng);
    let result = multilinear_sumcheck_partial(&mut v, &mut t, num_rounds, |_, _| {});
    assert_eq!(result.prover_messages.len(), num_rounds);
    assert_eq!(result.verifier_messages.len(), num_rounds);
    assert_eq!(v.len(), 1);
}

#[test]
fn test_partial_split_matches_full() {
    let num_vars = 8;
    let n = 1 << num_vars;
    let split_at = 3;

    let mut r = rng();
    let v_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut v_full = v_orig.clone();
    let mut full_rng = rng();
    let mut t_full = SanityTranscript::new(&mut full_rng);
    let full = multilinear_sumcheck(&mut v_full, &mut t_full, |_, _| {});

    let mut v = v_orig.clone();
    let mut split_rng = rng();
    let mut t_split = SanityTranscript::new(&mut split_rng);
    let first = multilinear_sumcheck_partial(&mut v, &mut t_split, split_at, |_, _| {});
    let second = multilinear_sumcheck_partial(&mut v, &mut t_split, num_vars - split_at, |_, _| {});

    let mut split_prover = first.prover_messages.clone();
    split_prover.extend(second.prover_messages.iter().copied());
    let mut split_verifier = first.verifier_messages.clone();
    split_verifier.extend(second.verifier_messages.iter().copied());

    assert_eq!(split_prover, full.prover_messages);
    assert_eq!(split_verifier, full.verifier_messages);
    assert_eq!(second.final_evaluation, full.final_evaluation);
    assert_eq!(first.final_evaluation, F64::ZERO);
}

#[test]
fn test_hook_called_once_per_round() {
    use std::cell::RefCell;
    let num_vars = 6;
    let n = 1 << num_vars;

    let mut r = rng();
    let mut v: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);

    let calls = RefCell::new(Vec::<usize>::new());
    let result = multilinear_sumcheck(&mut v, &mut t, |round, _| {
        calls.borrow_mut().push(round);
    });
    assert_eq!(result.prover_messages.len(), num_vars);
    assert_eq!(calls.into_inner(), (0..num_vars).collect::<Vec<_>>());
}

#[test]
fn test_zero_rounds_is_identity() {
    let mut r = rng();
    let v_orig: Vec<F64> = (0..8).map(|_| F64::rand(&mut r)).collect();
    let mut v = v_orig.clone();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);

    let result = multilinear_sumcheck_partial(&mut v, &mut t, 0, |_, _| {});
    assert!(result.prover_messages.is_empty());
    assert!(result.verifier_messages.is_empty());
    assert_eq!(v, v_orig);
}

#[test]
fn test_round0_msg_is_half_sums() {
    let n = 16_usize;
    let mut r = rng();
    let v: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut v_mut = v.clone();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let result = multilinear_sumcheck_partial(&mut v_mut, &mut t, 1, |_, _| {});
    let (s0, s1) = result.prover_messages[0];

    let half = n / 2;
    let expected_s0: F64 = v[..half].iter().copied().sum();
    let expected_s1: F64 = v[half..].iter().copied().sum();
    assert_eq!(s0, expected_s0);
    assert_eq!(s1, expected_s1);
}

#[test]
fn test_deterministic_under_same_seed() {
    let n = 1 << 5;
    let mut r = rng();
    let v_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let run = || -> _ {
        let mut v = v_orig.clone();
        let mut trng = rng();
        let mut t = SanityTranscript::new(&mut trng);
        multilinear_sumcheck(&mut v, &mut t, |_, _| {})
    };
    let r1 = run();
    let r2 = run();
    assert_eq!(r1.prover_messages, r2.prover_messages);
    assert_eq!(r1.verifier_messages, r2.verifier_messages);
    assert_eq!(r1.final_evaluation, r2.final_evaluation);
}

/// Reference: unfused half-split prover. Runs fold then compute each round.
fn reference_unfused(v_orig: &[F64]) -> Sumcheck<F64> {
    let mut v = v_orig.to_vec();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let num_rounds = if v.is_empty() {
        0
    } else {
        v.len().next_power_of_two().trailing_zeros() as usize
    };

    let mut prover_messages = Vec::with_capacity(num_rounds);
    let mut verifier_messages = Vec::with_capacity(num_rounds);
    let mut w: Option<F64> = None;

    for _ in 0..num_rounds {
        if let Some(weight) = w {
            fold_in_place(&mut v, weight);
        }
        let (s0, s1) = compute_ref(&v);
        prover_messages.push((s0, s1));
        t.send(s0);
        t.send(s1);
        let r: F64 = t.challenge();
        verifier_messages.push(r);
        w = Some(r);
    }
    if let Some(weight) = w {
        fold_in_place(&mut v, weight);
    }

    let final_evaluation = if v.len() == 1 { v[0] } else { F64::ZERO };
    Sumcheck {
        prover_messages,
        verifier_messages,
        final_evaluation,
    }
}

fn fold_in_place<F: Field>(values: &mut Vec<F>, weight: F) {
    if values.len() <= 1 {
        return;
    }
    let half = values.len().next_power_of_two() >> 1;
    let (low, high) = values.split_at_mut(half);
    let (low, tail) = low.split_at_mut(high.len());
    for (lo, hi) in low.iter_mut().zip(high.iter()) {
        *lo += (*hi - *lo) * weight;
    }
    for x in tail.iter_mut() {
        *x *= F::ONE - weight;
    }
    values.truncate(half);
}

fn compute_ref<F: Field>(values: &[F]) -> (F, F) {
    if values.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    if values.len() == 1 {
        return (values[0], F::ZERO);
    }
    let half = values.len().next_power_of_two() >> 1;
    let (lo, hi) = values.split_at(half);
    let (lo, lo_tail) = lo.split_at(hi.len());
    let mut s0 = F::ZERO;
    let mut s1 = F::ZERO;
    for (&l, &h) in lo.iter().zip(hi) {
        s0 += l;
        s1 += h;
    }
    let tail: F = lo_tail.iter().copied().sum();
    (s0 + tail, s1)
}

#[test]
fn test_fused_matches_unfused_reference_pow2() {
    for &num_vars in &[1_usize, 2, 4, 7, 10] {
        let n = 1 << num_vars;
        let mut r = rng();
        let v_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

        let ref_result = reference_unfused(&v_orig);

        let mut v = v_orig.clone();
        let mut trng = rng();
        let mut t = SanityTranscript::new(&mut trng);
        let fused = multilinear_sumcheck(&mut v, &mut t, |_, _| {});

        assert_eq!(fused.prover_messages, ref_result.prover_messages, "n={n}");
        assert_eq!(
            fused.verifier_messages, ref_result.verifier_messages,
            "n={n}"
        );
        assert_eq!(fused.final_evaluation, ref_result.final_evaluation, "n={n}");
    }
}

#[test]
fn test_fused_matches_unfused_reference_non_pow2() {
    for &n in &[3_usize, 5, 13, 33, 100] {
        let mut r = rng();
        let v_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

        let ref_result = reference_unfused(&v_orig);

        let mut v = v_orig.clone();
        let mut trng = rng();
        let mut t = SanityTranscript::new(&mut trng);
        let fused = multilinear_sumcheck(&mut v, &mut t, |_, _| {});

        assert_eq!(fused.prover_messages, ref_result.prover_messages, "n={n}");
        assert_eq!(
            fused.verifier_messages, ref_result.verifier_messages,
            "n={n}"
        );
        assert_eq!(fused.final_evaluation, ref_result.final_evaluation, "n={n}");
    }
}

// Silence unused-import warning when built without tests touching AdditiveGroup.
const _: F64 = <F64 as AdditiveGroup>::ZERO;
