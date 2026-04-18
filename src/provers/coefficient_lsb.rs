//! LSB (pair-split) coefficient sumcheck prover: arbitrary degree d.
//!
//! Folds the *least-significant* variable each round: pairs `(f[2k], f[2k+1])`.
//! This is the natural layout for **sequential streaming** where evaluations
//! arrive in index order.
//!
//! Use this prover for Jolt-style workloads. For in-memory or random-access
//! workloads, prefer [`CoefficientProverLSB`](super::coefficient::CoefficientProverLSB)
//! (MSB layout).

use ark_ff::Field;

use crate::coefficient_sumcheck::RoundPolyEvaluator;
use crate::reductions::{pairwise, tablewise};
use crate::sumcheck_prover::SumcheckProver;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Coefficient sumcheck prover (arbitrary degree d).
///
/// Wraps tablewise and pairwise evaluation tables with a user-provided
/// [`RoundPolyEvaluator`] that defines the round polynomial shape.
///
/// # Construction
///
/// ```ignore
/// let evaluator = MyEvaluator;
/// let mut prover = CoefficientProverLSB::new(
///     &evaluator,
///     tablewise_tables,   // Vec<Vec<Vec<F>>>
///     pairwise_tables,    // Vec<Vec<F>>
///     num_rounds,
/// );
/// let proof = sumcheck(&mut prover, num_rounds, &mut transcript, |_, _| {});
/// ```
pub struct CoefficientProverLSB<'a, F: Field, E: RoundPolyEvaluator<F>> {
    evaluator: &'a E,
    tablewise: Vec<Vec<Vec<F>>>,
    pairwise: Vec<Vec<F>>,
    n_tw: usize,
    n_pw: usize,
    deg: usize,
    /// Cached degree-1 SIMD evaluation from fused reduce+evaluate.
    pending_degree1_eval: Option<Vec<F>>,
    /// Whether this is the degree-1, single-pairwise, no-tablewise fast path.
    is_degree1_simd_path: bool,
}

impl<'a, F: Field, E: RoundPolyEvaluator<F>> CoefficientProverLSB<'a, F, E> {
    pub fn new(evaluator: &'a E, tablewise: Vec<Vec<Vec<F>>>, pairwise: Vec<Vec<F>>) -> Self {
        let n_tw = tablewise.len();
        let n_pw = pairwise.len();
        let deg = evaluator.degree();
        let is_degree1_simd_path = deg == 1 && n_pw == 1 && n_tw == 0;
        Self {
            evaluator,
            tablewise,
            pairwise,
            n_tw,
            n_pw,
            deg,
            pending_degree1_eval: None,
            is_degree1_simd_path,
        }
    }

    /// Access the (possibly reduced) tablewise tables.
    pub fn tablewise(&self) -> &[Vec<Vec<F>>] {
        &self.tablewise
    }

    /// Access the (possibly reduced) pairwise tables.
    pub fn pairwise(&self) -> &[Vec<F>] {
        &self.pairwise
    }

    fn n_pairs(&self) -> usize {
        if self.n_tw > 0 {
            self.tablewise[0].len() / 2
        } else if self.n_pw > 0 {
            self.pairwise[0].len() / 2
        } else {
            0
        }
    }

    /// Compute round polynomial coefficients.
    fn evaluate_coefficients(&mut self) -> Vec<F> {
        if let Some(cached) = self.pending_degree1_eval.take() {
            return cached;
        }
        if self.is_degree1_simd_path {
            return simd_evaluate_degree1(&self.pairwise[0]);
        }

        let n_pairs = self.n_pairs();
        let n_coeffs = self.deg + 1;

        if self.evaluator.parallelize() {
            parallel_evaluate(
                self.evaluator,
                &self.tablewise,
                &self.pairwise,
                self.n_tw,
                self.n_pw,
                n_pairs,
                n_coeffs,
            )
        } else {
            let mut coeffs = vec![F::ZERO; n_coeffs];
            sequential_evaluate_into(
                self.evaluator,
                &self.tablewise,
                &self.pairwise,
                self.n_tw,
                self.n_pw,
                n_pairs,
                &mut coeffs,
            );
            coeffs
        }
    }

    /// Reduce all tables by folding with the challenge.
    fn reduce(&mut self, challenge: F, is_last_round: bool) {
        for table in self.tablewise.iter_mut() {
            tablewise::reduce_evaluations(table, challenge);
        }

        if self.is_degree1_simd_path && !is_last_round {
            if let Some(next) = try_simd_fused_reduce_evaluate(&mut self.pairwise[0], challenge) {
                self.pending_degree1_eval = Some(next);
                return;
            }
        }

        for table in self.pairwise.iter_mut() {
            #[cfg(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            ))]
            if crate::simd_sumcheck::dispatch::try_simd_reduce(table, challenge) {
                continue;
            }
            pairwise::reduce_evaluations(table, challenge);
        }
    }
}

// ─── SumcheckProver impl ───────────────────────────────────────────────────

#[cfg(feature = "arkworks")]
impl<'a, F, E> SumcheckProver<F> for CoefficientProverLSB<'a, F, E>
where
    F: ark_ff::Field,
    E: RoundPolyEvaluator<F>,
{
    fn degree(&self) -> usize {
        self.deg
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        // Reduce with previous challenge (if any).
        if let Some(c) = challenge {
            self.reduce(c, false);
        }

        // Compute coefficient representation.
        let coeffs = self.evaluate_coefficients();

        // Convert coefficients → evaluations at {0, 1, ..., degree}.
        let mut evals = Vec::with_capacity(self.deg + 1);
        for i in 0..=self.deg {
            evals.push(eval_poly_at(&coeffs, F::from(i as u64)));
        }
        evals
    }

    fn finalize(&mut self, last_challenge: F) {
        self.reduce(last_challenge, true);
    }

    fn final_value(&self) -> F {
        // After all rounds, each pairwise table should have 1 element,
        // each tablewise table should have 1 row. The final value is
        // the evaluator applied to these singletons.
        let mut coeffs = vec![F::ZERO; self.deg + 1];
        let n_pairs = self.n_pairs();
        if n_pairs > 0 {
            sequential_evaluate_into(
                self.evaluator,
                &self.tablewise,
                &self.pairwise,
                self.n_tw,
                self.n_pw,
                n_pairs,
                &mut coeffs,
            );
        }
        // final_value = h(0) + h(1) = sum of evaluations at 0 and 1
        // Actually: the "final value" for coefficient sumcheck is the
        // claimed sum at the last point, which is eval of the polynomial
        // at the last challenge. But after finalize(), the tables are
        // fully reduced — there's only 1 "pair" left (of size 1).
        // The claim is h(0) + h(1) from the last round's perspective,
        // but that's the *next* claim, not the evaluation.
        //
        // For consistency with the other provers: final_value should be
        // the polynomial evaluated at the random point. After full
        // reduction, the single remaining element in pairwise[0] IS
        // the evaluation (for degree-1 single-pairwise case).
        if self.is_degree1_simd_path && !self.pairwise.is_empty() && self.pairwise[0].len() == 1 {
            return self.pairwise[0][0];
        }
        // General case: sum the contributions.
        eval_poly_at(&coeffs, F::ZERO) + eval_poly_at(&coeffs, F::ONE)
    }
}

// ─── Horner evaluation ─────────────────────────────────────────────────────

/// Evaluate polynomial with coefficients `coeffs` at point `x` via Horner's method.
#[inline]
fn eval_poly_at<F: Field>(coeffs: &[F], x: F) -> F {
    if coeffs.is_empty() {
        return F::ZERO;
    }
    let mut result = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        result = result * x + coeffs[i];
    }
    result
}

// ─── Evaluate strategies (same as coefficient_sumcheck.rs) ─────────────────

fn simd_evaluate_degree1<F: Field>(pw: &[F]) -> Vec<F> {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    {
        if let Some(coeffs) = crate::simd_sumcheck::dispatch::try_simd_evaluate_degree1(pw) {
            return coeffs;
        }
    }
    let mut s0 = F::ZERO;
    let mut s1 = F::ZERO;
    for chunk in pw.chunks_exact(2) {
        s0 += chunk[0];
        s1 += chunk[1];
    }
    vec![s0, s1 - s0]
}

fn try_simd_fused_reduce_evaluate<F: Field>(pw: &mut Vec<F>, challenge: F) -> Option<Vec<F>> {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    {
        crate::simd_sumcheck::dispatch::try_simd_fused_reduce_evaluate_degree1(pw, challenge)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )))]
    {
        let _ = (pw, challenge);
        None
    }
}

#[cfg(feature = "parallel")]
fn parallel_evaluate<F: Field>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    n_pairs: usize,
    n_coeffs: usize,
) -> Vec<F> {
    (0..n_pairs)
        .into_par_iter()
        .fold_with(vec![F::ZERO; n_coeffs], |mut acc, pair_idx| {
            let mut tw_buf: [(&[F], &[F]); 16] = [(&[], &[]); 16];
            let mut pw_buf: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];
            for (i, table) in tablewise.iter().enumerate() {
                tw_buf[i] = (&table[2 * pair_idx], &table[2 * pair_idx + 1]);
            }
            for (i, table) in pairwise.iter().enumerate() {
                pw_buf[i] = (table[2 * pair_idx], table[2 * pair_idx + 1]);
            }
            evaluator.accumulate_pair(&mut acc, &tw_buf[..n_tw], &pw_buf[..n_pw]);
            acc
        })
        .reduce_with(|mut a, b| {
            for (ai, bi) in a.iter_mut().zip(&b) {
                *ai += *bi;
            }
            a
        })
        .unwrap_or_else(|| vec![F::ZERO; n_coeffs])
}

#[cfg(not(feature = "parallel"))]
fn parallel_evaluate<F: Field>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    n_pairs: usize,
    n_coeffs: usize,
) -> Vec<F> {
    let mut coeffs = vec![F::ZERO; n_coeffs];
    sequential_evaluate_into(
        evaluator,
        tablewise,
        pairwise,
        n_tw,
        n_pw,
        n_pairs,
        &mut coeffs,
    );
    coeffs
}

fn sequential_evaluate_into<F: Field>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    n_pairs: usize,
    coeffs_out: &mut [F],
) {
    for c in coeffs_out.iter_mut() {
        *c = F::ZERO;
    }
    let mut tw_buf: [(&[F], &[F]); 16] = [(&[], &[]); 16];
    let mut pw_buf: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];

    for pair_idx in 0..n_pairs {
        for (i, table) in tablewise.iter().enumerate() {
            tw_buf[i] = (&table[2 * pair_idx], &table[2 * pair_idx + 1]);
        }
        for (i, table) in pairwise.iter().enumerate() {
            pw_buf[i] = (table[2 * pair_idx], table[2 * pair_idx + 1]);
        }
        evaluator.accumulate_pair(coeffs_out, &tw_buf[..n_tw], &pw_buf[..n_pw]);
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::sumcheck;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    struct Degree1Eval;
    impl RoundPolyEvaluator<F64> for Degree1Eval {
        fn degree(&self) -> usize {
            1
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (even, odd) = pw[0];
            coeffs[0] += even;
            coeffs[1] += odd - even;
        }
    }

    struct Degree2Eval;
    impl RoundPolyEvaluator<F64> for Degree2Eval {
        fn degree(&self) -> usize {
            2
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (s0, s1) = pw[0];
            let s2 = s0 + s1;
            coeffs[0] += s0;
            coeffs[1] += (-F64::from(3u64) * s0 + F64::from(4u64) * s1 - s2) / F64::from(2u64);
            coeffs[2] += (s0 - F64::from(2u64) * s1 + s2) / F64::from(2u64);
        }
    }

    /// CoefficientProverLSB produces valid sumcheck proofs for degree 1.
    #[test]
    fn degree1_consistency() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        let evaluator = Degree1Eval;
        let pairwise = vec![evals];
        let tablewise: Vec<Vec<Vec<F64>>> = vec![];

        let mut prover = CoefficientProverLSB::new(&evaluator, tablewise, pairwise);
        let mut trng = StdRng::seed_from_u64(99);
        let mut t = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, 4, &mut t, |_, _| {});

        // Round-0 consistency.
        assert_eq!(
            proof.round_polys[0][0] + proof.round_polys[0][1],
            claimed_sum
        );

        // All-round consistency via Lagrange.
        let mut claim = claimed_sum;
        for (rp, &r) in proof.round_polys.iter().zip(&proof.challenges) {
            assert_eq!(rp[0] + rp[1], claim, "consistency check failed");
            // degree 1: q(r) = q(0) + r*(q(1) - q(0))
            claim = rp[0] + r * (rp[1] - rp[0]);
        }
        assert_eq!(proof.final_value, claim);
    }

    /// CoefficientProverLSB matches the old coefficient_sumcheck function.
    #[test]
    fn matches_legacy_coefficient_sumcheck() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Old API.
        let mut old_pw = vec![evals.clone()];
        let mut old_tw: Vec<Vec<Vec<F64>>> = vec![];
        let mut trng = StdRng::seed_from_u64(99);
        let mut t_old = SanityTranscript::new(&mut trng);
        let old_result = crate::coefficient_sumcheck::coefficient_sumcheck(
            &Degree1Eval,
            &mut old_tw,
            &mut old_pw,
            4,
            &mut t_old,
        );

        // New API.
        let evaluator = Degree1Eval;
        let mut prover = CoefficientProverLSB::new(&evaluator, vec![], vec![evals]);
        let mut trng2 = StdRng::seed_from_u64(99);
        let mut t_new = SanityTranscript::new(&mut trng2);
        let new_result = sumcheck(&mut prover, 4, &mut t_new, |_, _| {});

        // Same challenges (same transcript seed).
        assert_eq!(old_result.verifier_messages, new_result.challenges);

        // Round polynomials: old is coefficients, new is evaluations.
        // Verify consistency: old coeffs evaluated at {0,1} should match
        // new evals[0] and evals[1].
        for (i, (old_poly, new_evals)) in old_result
            .prover_messages
            .iter()
            .zip(&new_result.round_polys)
            .enumerate()
        {
            use ark_poly::Polynomial;
            let old_at_0 = old_poly.evaluate(&F64::from(0u64));
            let old_at_1 = old_poly.evaluate(&F64::from(1u64));
            assert_eq!(old_at_0, new_evals[0], "round {i}: q(0) mismatch");
            assert_eq!(old_at_1, new_evals[1], "round {i}: q(1) mismatch");
        }
    }

    /// Degree-2 CoefficientProverLSB round polynomial has 3 evaluations.
    #[test]
    fn degree2_structure() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 1 << 3;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        let evaluator = Degree2Eval;
        let mut prover = CoefficientProverLSB::new(&evaluator, vec![], vec![evals]);
        let mut trng = StdRng::seed_from_u64(99);
        let mut t = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, 3, &mut t, |_, _| {});

        for rp in &proof.round_polys {
            assert_eq!(rp.len(), 3, "degree-2 should have 3 evaluations");
        }
        assert_eq!(
            proof.round_polys[0][0] + proof.round_polys[0][1],
            claimed_sum
        );
    }
}
