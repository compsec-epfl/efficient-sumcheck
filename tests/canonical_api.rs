//! Integration tests for the canonical sumcheck API (Thaler §4.1).
//!
//! Tests the `SumcheckProver` trait, `sumcheck()` runner,
//! `MultilinearProver`, and `InnerProductProver`.

use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use efficient_sumcheck::provers::inner_product::InnerProductProver;
use efficient_sumcheck::provers::multilinear::MultilinearProver;
use efficient_sumcheck::runner::sumcheck;
use efficient_sumcheck::tests::F64;
use efficient_sumcheck::transcript::SanityTranscript;

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

/// Evaluate degree-d polynomial from evaluations at {0,1,...,d} at point r
/// via Lagrange interpolation.
fn lagrange_eval(evals: &[F64], r: F64) -> F64 {
    let d = evals.len();
    let mut result = F64::ZERO;
    for i in 0..d {
        let mut basis = <F64 as Field>::ONE;
        for j in 0..d {
            if j != i {
                let ni = F64::from(i as u64);
                let nj = F64::from(j as u64);
                basis *= (r - nj) / (ni - nj);
            }
        }
        result += evals[i] * basis;
    }
    result
}

/// Verify a SumcheckProof by checking consistency equations.
fn verify_proof(
    claimed_sum: F64,
    round_polys: &[Vec<F64>],
    challenges: &[F64],
    final_value: F64,
) {
    let num_rounds = round_polys.len();
    assert_eq!(challenges.len(), num_rounds);

    let mut claim = claimed_sum;
    for (j, (rp, &r)) in round_polys.iter().zip(challenges).enumerate() {
        // q_j(0) + q_j(1) == claim
        let sum_01 = rp[0] + rp[1];
        assert_eq!(sum_01, claim, "round {j}: consistency check failed");
        // Update claim = q_j(r_j).
        claim = lagrange_eval(rp, r);
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
        assert_eq!(rp.len(), 2, "degree-1 round poly should have 2 evaluations");
    }

    // Verify consistency.
    verify_proof(claimed_sum, &proof.round_polys, &proof.challenges, proof.final_value);

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
        assert_eq!(rp.len(), 3, "degree-2 round poly should have 3 evaluations");
    }

    // Verify consistency.
    verify_proof(claimed_sum, &proof.round_polys, &proof.challenges, proof.final_value);

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
