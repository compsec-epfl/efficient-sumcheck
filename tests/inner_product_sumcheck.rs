//! Integration tests for the MSB fused inner-product sumcheck.

use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use effsc::tests::F64;
use effsc::transcript::{ProverTranscript, SanityTranscript};
use effsc::{inner_product_sumcheck, inner_product_sumcheck_partial, ProductSumcheck};

const SEED: u64 = 0xA110C8ED;

fn rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

/// Evaluate the multilinear extension of `evals` at `point` with MSB
/// ordering (pop the top half each round).
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

    let mut prover_rng = rng();
    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut t_prove = SanityTranscript::new(&mut prover_rng);
    let result: ProductSumcheck<F64> =
        inner_product_sumcheck(&mut a, &mut b, &mut t_prove, |_, _| {});

    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(result.prover_messages.len(), num_vars);
    assert_eq!(result.verifier_messages.len(), num_vars);
    assert_eq!(result.final_evaluations, (a[0], b[0]));

    // Folded values match an independent MLE evaluation at the challenge point.
    assert_eq!(multilinear_extend(&a_orig, &result.verifier_messages), a[0]);
    assert_eq!(multilinear_extend(&b_orig, &result.verifier_messages), b[0]);
}

#[test]
fn test_non_power_of_two_partial_runs() {
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
    let result = inner_product_sumcheck_partial(&mut a, &mut b, &mut t, num_rounds, |_, _| {});
    assert_eq!(result.prover_messages.len(), num_rounds);
    assert_eq!(result.verifier_messages.len(), num_rounds);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
}

#[test]
fn test_partial_split_matches_full() {
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
    let full = inner_product_sumcheck(&mut a_full, &mut b_full, &mut t_full, |_, _| {});

    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut split_rng = rng();
    let mut t_split = SanityTranscript::new(&mut split_rng);
    let first = inner_product_sumcheck_partial(&mut a, &mut b, &mut t_split, split_at, |_, _| {});
    let second = inner_product_sumcheck_partial(
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
    let result = inner_product_sumcheck(&mut a, &mut b, &mut t, |round, _| {
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

    let result = inner_product_sumcheck_partial(&mut a, &mut b, &mut t, 0, |_, _| {});
    assert!(result.prover_messages.is_empty());
    assert!(result.verifier_messages.is_empty());
    assert_eq!(a, a_orig);
    assert_eq!(b, b_orig);
}

#[test]
fn test_prover_msg_is_difference_form() {
    // Round-0 (c0, c2) in difference form:
    //   c0 = Σ a_lo · b_lo                    (= q(0))
    //   c2 = Σ (a_hi − a_lo)·(b_hi − b_lo)    (= [x²] q(x))
    let n = 16_usize;
    let mut r = rng();
    let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut a_mut = a.clone();
    let mut b_mut = b.clone();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let result = inner_product_sumcheck_partial(&mut a_mut, &mut b_mut, &mut t, 1, |_, _| {});
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
    let n = 1 << 5;
    let mut r = rng();
    let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let run = || -> _ {
        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        let mut trng = rng();
        let mut t = SanityTranscript::new(&mut trng);
        inner_product_sumcheck(&mut a, &mut b, &mut t, |_, _| {})
    };
    let r1 = run();
    let r2 = run();
    assert_eq!(r1.prover_messages, r2.prover_messages);
    assert_eq!(r1.verifier_messages, r2.verifier_messages);
    assert_eq!(r1.final_evaluations, r2.final_evaluations);
}

/// Reference unfused half-split prover. Runs the protocol by folding then
/// computing each round with plain scalar loops. Transcript must match the
/// fused path bit-for-bit.
fn reference_unfused(a_orig: &[F64], b_orig: &[F64]) -> ProductSumcheck<F64> {
    let mut a = a_orig.to_vec();
    let mut b = b_orig.to_vec();
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let num_rounds = if a.is_empty() {
        0
    } else {
        a.len().next_power_of_two().trailing_zeros() as usize
    };

    let mut prover_messages = Vec::with_capacity(num_rounds);
    let mut verifier_messages = Vec::with_capacity(num_rounds);
    let mut w: Option<F64> = None;

    for _ in 0..num_rounds {
        if let Some(weight) = w {
            fold_in_place(&mut a, weight);
            fold_in_place(&mut b, weight);
        }
        let (c0, c2) = compute_ref(&a, &b);
        prover_messages.push((c0, c2));
        t.send(c0);
        t.send(c2);
        let r: F64 = t.challenge();
        verifier_messages.push(r);
        w = Some(r);
    }
    if let Some(weight) = w {
        fold_in_place(&mut a, weight);
        fold_in_place(&mut b, weight);
    }

    let final_evaluations = if a.len() == 1 {
        (a[0], b[0])
    } else {
        (F64::ZERO, F64::ZERO)
    };
    ProductSumcheck {
        prover_messages,
        verifier_messages,
        final_evaluations,
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

fn compute_ref<F: Field>(a: &[F], b: &[F]) -> (F, F) {
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
    let (a0, a0_tail) = a0.split_at(a1.len());
    let (b0, b0_tail) = b0.split_at(a1.len());
    let mut c0 = F::ZERO;
    let mut c2 = F::ZERO;
    for ((&x0, &x1), (&y0, &y1)) in a0.iter().zip(a1).zip(b0.iter().zip(b1)) {
        c0 += x0 * y0;
        c2 += (x1 - x0) * (y1 - y0);
    }
    let tail: F = a0_tail.iter().zip(b0_tail).map(|(x, y)| *x * *y).sum();
    (c0 + tail, c2 + tail)
}

#[test]
fn test_fused_matches_unfused_reference_pow2() {
    for &num_vars in &[1_usize, 2, 4, 7, 10] {
        let n = 1 << num_vars;
        let mut r = rng();
        let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
        let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

        let ref_result = reference_unfused(&a_orig, &b_orig);

        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        let mut trng = rng();
        let mut t = SanityTranscript::new(&mut trng);
        let fused = inner_product_sumcheck(&mut a, &mut b, &mut t, |_, _| {});

        assert_eq!(fused.prover_messages, ref_result.prover_messages, "n={n}");
        assert_eq!(
            fused.verifier_messages, ref_result.verifier_messages,
            "n={n}"
        );
        assert_eq!(
            fused.final_evaluations, ref_result.final_evaluations,
            "n={n}"
        );
    }
}

#[test]
fn test_fused_matches_unfused_reference_non_pow2() {
    for &n in &[3_usize, 5, 13, 33, 100] {
        let mut r = rng();
        let a_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
        let b_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

        let ref_result = reference_unfused(&a_orig, &b_orig);

        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        let mut trng = rng();
        let mut t = SanityTranscript::new(&mut trng);
        let fused = inner_product_sumcheck(&mut a, &mut b, &mut t, |_, _| {});

        assert_eq!(fused.prover_messages, ref_result.prover_messages, "n={n}");
        assert_eq!(
            fused.verifier_messages, ref_result.verifier_messages,
            "n={n}"
        );
        assert_eq!(
            fused.final_evaluations, ref_result.final_evaluations,
            "n={n}"
        );
    }
}

// Silence unused-import warning when built without tests touching AdditiveGroup.
const _: F64 = <F64 as AdditiveGroup>::ZERO;
