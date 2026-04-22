//! Integration tests for the canonical sumcheck API (Thaler §4.1).
//!
//! Tests the `SumcheckProver` trait, `sumcheck()` runner,
//! `MultilinearProver`, and `InnerProductProver`.

use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use effsc::provers::inner_product::InnerProductProver;
use effsc::provers::multilinear::MultilinearProver;
use effsc::runner::sumcheck;
use effsc::tests::F64;
use effsc::transcript::SanityTranscript;

const SEED: u64 = 0xDEAD_BEEF;

fn rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

/// Independent MLE evaluation (MSB half-split fold).
fn mle_eval(evals: &[F64], point: &[F64]) -> F64 {
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

/// Reconstruct the degree-`d` round polynomial's evaluation at `r` from an
/// EvalsInfty-format wire message.
///
/// Wire layout: `[h(0), h(∞), h(2), ..., h(d-1)]`, with `h(1) = claim - h(0)`
/// derived from the consistency check.
fn evalsinfty_eval(wire: &[F64], claim: F64, d: usize, r: F64) -> F64 {
    if d == 0 {
        return wire[0];
    }
    let h0 = wire[0];
    let h1 = claim - h0;
    let h_inf = if d >= 2 { wire[1] } else { h1 - h0 };

    // Build (h_i - h_inf·i^d) at i=0..d-1, then Lagrange-interpolate
    // the degree-(d-1) remainder at r, then add h_inf·r^d.
    let pow = |x: F64, k: usize| -> F64 {
        let mut v = <F64 as Field>::ONE;
        for _ in 0..k {
            v *= x;
        }
        v
    };
    let mut finite = Vec::with_capacity(d);
    finite.push(h0);
    if d >= 1 {
        finite.push(h1 - h_inf);
    }
    for i in 2..d {
        let hi = wire[i];
        let i_f = F64::from(i as u64);
        finite.push(hi - h_inf * pow(i_f, d));
    }

    // Lagrange over {0, 1, ..., d-1}.
    let mut q_r = F64::ZERO;
    let n = finite.len();
    for i in 0..n {
        let mut basis = <F64 as Field>::ONE;
        for j in 0..n {
            if j != i {
                let ni = F64::from(i as u64);
                let nj = F64::from(j as u64);
                basis *= (r - nj) / (ni - nj);
            }
        }
        q_r += finite[i] * basis;
    }

    q_r + h_inf * pow(r, d)
}

/// Verify a SumcheckProof by replaying the round reductions under the
/// EvalsInfty wire format.
fn verify_proof(
    claimed_sum: F64,
    round_polys: &[Vec<F64>],
    challenges: &[F64],
    final_value: F64,
    degree: usize,
) {
    let num_rounds = round_polys.len();
    assert_eq!(challenges.len(), num_rounds);

    let mut claim = claimed_sum;
    for (rp, &r) in round_polys.iter().zip(challenges) {
        claim = evalsinfty_eval(rp, claim, degree, r);
    }
    assert_eq!(claim, final_value, "final value mismatch");
}

// ─── MultilinearProver ─────────────────────────────────────────────────────

#[test]
fn multilinear_full_roundtrip() {
    let num_vars = 8;
    let n = 1 << num_vars;

    let mut r = rng();
    let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let claimed_sum: F64 = evals.iter().copied().sum();

    let mut prover = MultilinearProver::new(evals.clone());
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let proof = sumcheck(&mut prover, num_vars, &mut t, |_, _| {});

    // Proof structure.
    assert_eq!(proof.round_polys.len(), num_vars);
    assert_eq!(proof.challenges.len(), num_vars);
    for rp in &proof.round_polys {
        assert_eq!(rp.len(), 1, "degree-1 EvalsInfty round poly has 1 value");
    }

    // Verify consistency.
    verify_proof(
        claimed_sum,
        &proof.round_polys,
        &proof.challenges,
        proof.final_value,
        1,
    );

    // Final value matches independent MLE evaluation.
    assert_eq!(proof.final_value, mle_eval(&evals, &proof.challenges));

    // Prover post-state.
    assert_eq!(prover.evals().len(), 1);
    assert_eq!(prover.evals()[0], proof.final_value);
}

#[test]
fn multilinear_partial_then_continue() {
    let num_vars = 8;
    let split = 3;
    let n = 1 << num_vars;

    let mut r = rng();
    let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    // Full.
    let mut prover_full = MultilinearProver::new(evals.clone());
    let mut trng = rng();
    let mut t_full = SanityTranscript::new(&mut trng);
    let full = sumcheck(&mut prover_full, num_vars, &mut t_full, |_, _| {});

    // Split.
    let mut prover = MultilinearProver::new(evals);
    let mut trng2 = rng();
    let mut t_split = SanityTranscript::new(&mut trng2);
    let first = sumcheck(&mut prover, split, &mut t_split, |_, _| {});
    let second = sumcheck(&mut prover, num_vars - split, &mut t_split, |_, _| {});

    // Round polys concatenate to match full.
    let mut combined_polys = first.round_polys.clone();
    combined_polys.extend(second.round_polys.iter().cloned());
    assert_eq!(combined_polys, full.round_polys);

    let mut combined_challenges = first.challenges.clone();
    combined_challenges.extend(second.challenges.iter().copied());
    assert_eq!(combined_challenges, full.challenges);

    assert_eq!(second.final_value, full.final_value);
}

#[test]
fn multilinear_hook_fires() {
    use std::cell::RefCell;
    let num_vars = 5;
    let n = 1 << num_vars;

    let mut r = rng();
    let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let mut prover = MultilinearProver::new(evals);
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);

    let calls = RefCell::new(Vec::<usize>::new());
    let _ = sumcheck(&mut prover, num_vars, &mut t, |round, _| {
        calls.borrow_mut().push(round);
    });
    assert_eq!(calls.into_inner(), (0..num_vars).collect::<Vec<_>>());
}

#[test]
fn multilinear_non_pow2() {
    let n = 13_usize;
    let num_rounds = n.next_power_of_two().trailing_zeros() as usize;

    let mut r = rng();
    let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    let mut prover = MultilinearProver::new(evals);
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let proof = sumcheck(&mut prover, num_rounds, &mut t, |_, _| {});

    assert_eq!(proof.round_polys.len(), num_rounds);
    assert_eq!(prover.evals().len(), 1);
}

// ─── InnerProductProver ────────────────────────────────────────────────────

#[test]
fn inner_product_full_roundtrip() {
    let num_vars = 8;
    let n = 1 << num_vars;

    let mut r = rng();
    let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let claimed_sum: F64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();

    let mut prover = InnerProductProver::new(a.clone(), b.clone());
    let mut trng = rng();
    let mut t = SanityTranscript::new(&mut trng);
    let proof = sumcheck(&mut prover, num_vars, &mut t, |_, _| {});

    // Proof structure.
    assert_eq!(proof.round_polys.len(), num_vars);
    assert_eq!(proof.challenges.len(), num_vars);
    for rp in &proof.round_polys {
        assert_eq!(rp.len(), 2, "degree-2 EvalsInfty round poly has 2 values");
    }

    // Verify consistency.
    verify_proof(
        claimed_sum,
        &proof.round_polys,
        &proof.challenges,
        proof.final_value,
        2,
    );

    // Final value == f(r) * g(r).
    let (fa, fb) = prover.final_evaluations();
    assert_eq!(proof.final_value, fa * fb);

    // Independent MLE check.
    assert_eq!(fa, mle_eval(&a, &proof.challenges));
    assert_eq!(fb, mle_eval(&b, &proof.challenges));
}

#[test]
fn inner_product_partial_then_continue() {
    let num_vars = 8;
    let split = 3;
    let n = 1 << num_vars;

    let mut r = rng();
    let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();
    let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut r)).collect();

    // Full.
    let mut prover_full = InnerProductProver::new(a.clone(), b.clone());
    let mut trng = rng();
    let mut t_full = SanityTranscript::new(&mut trng);
    let full = sumcheck(&mut prover_full, num_vars, &mut t_full, |_, _| {});

    // Split.
    let mut prover = InnerProductProver::new(a, b);
    let mut trng2 = rng();
    let mut t_split = SanityTranscript::new(&mut trng2);
    let first = sumcheck(&mut prover, split, &mut t_split, |_, _| {});
    let second = sumcheck(&mut prover, num_vars - split, &mut t_split, |_, _| {});

    let mut combined_polys = first.round_polys.clone();
    combined_polys.extend(second.round_polys.iter().cloned());
    assert_eq!(combined_polys, full.round_polys);

    let mut combined_challenges = first.challenges.clone();
    combined_challenges.extend(second.challenges.iter().copied());
    assert_eq!(combined_challenges, full.challenges);

    assert_eq!(second.final_value, full.final_value);
}
