//! MSB (half-split) coefficient sumcheck prover: arbitrary degree d.
//!
//! Folds the *most-significant* variable each round: pairs `(f[k], f[k+L/2])`.
//! This is optimal for in-memory and random-access-streaming workloads.
//!
//! For sequential streaming (Jolt-style), use
//! [`CoefficientProverLSB`](super::coefficient_lsb::CoefficientProverLSB).

use ark_ff::Field;

use crate::coefficient_sumcheck::RoundPolyEvaluator;
use crate::sumcheck_prover::SumcheckProver;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// MSB coefficient sumcheck prover (arbitrary degree d, half-split layout).
pub struct CoefficientProver<'a, F: Field, E: RoundPolyEvaluator<F>> {
    evaluator: &'a E,
    tablewise: Vec<Vec<Vec<F>>>,
    pairwise: Vec<Vec<F>>,
    n_tw: usize,
    n_pw: usize,
    deg: usize,
}

impl<'a, F: Field, E: RoundPolyEvaluator<F>> CoefficientProver<'a, F, E> {
    pub fn new(evaluator: &'a E, tablewise: Vec<Vec<Vec<F>>>, pairwise: Vec<Vec<F>>) -> Self {
        let n_tw = tablewise.len();
        let n_pw = pairwise.len();
        let deg = evaluator.degree();
        Self {
            evaluator,
            tablewise,
            pairwise,
            n_tw,
            n_pw,
            deg,
        }
    }

    pub fn tablewise(&self) -> &[Vec<Vec<F>>] {
        &self.tablewise
    }

    pub fn pairwise(&self) -> &[Vec<F>] {
        &self.pairwise
    }

    fn half(&self) -> usize {
        if self.n_tw > 0 {
            self.tablewise[0].len() / 2
        } else if self.n_pw > 0 {
            self.pairwise[0].len() / 2
        } else {
            0
        }
    }

    /// Compute round polynomial coefficients using MSB (half-split) pairing.
    fn evaluate_coefficients(&self) -> Vec<F> {
        let half = self.half();
        let n_coeffs = self.deg + 1;

        if self.evaluator.parallelize() && half > 0 {
            msb_parallel_evaluate(
                self.evaluator,
                &self.tablewise,
                &self.pairwise,
                self.n_tw,
                self.n_pw,
                half,
                n_coeffs,
            )
        } else {
            let mut coeffs = vec![F::ZERO; n_coeffs];
            msb_sequential_evaluate_into(
                self.evaluator,
                &self.tablewise,
                &self.pairwise,
                self.n_tw,
                self.n_pw,
                half,
                &mut coeffs,
            );
            coeffs
        }
    }

    /// MSB half-split reduce: `new[k] = v[k] + c*(v[k+half] - v[k])`.
    fn reduce(&mut self, challenge: F) {
        // Pairwise tables: MSB fold.
        for table in self.pairwise.iter_mut() {
            #[cfg(all(
                feature = "simd",
                any(
                    target_arch = "aarch64",
                    all(target_arch = "x86_64", target_feature = "avx512ifma")
                )
            ))]
            if crate::simd_sumcheck::dispatch::try_simd_reduce_msb(table, challenge) {
                continue;
            }
            msb_fold_vec(table, challenge);
        }
        // Tablewise tables: MSB fold each row-vector.
        for table in self.tablewise.iter_mut() {
            msb_fold_tablewise(table, challenge);
        }
    }
}

// ─── MSB fold helpers ──────────────────────────────────────────────────────

/// In-place MSB fold for a flat vector: `new[k] = v[k] + c*(v[k+half] - v[k])`.
fn msb_fold_vec<F: Field>(v: &mut Vec<F>, challenge: F) {
    if v.len() <= 1 {
        return;
    }
    let half = v.len() / 2;
    for k in 0..half {
        v[k] = v[k] + challenge * (v[k + half] - v[k]);
    }
    v.truncate(half);
}

/// MSB fold for tablewise: each row-vector is folded by pairing
/// `(table[k], table[k+half])` and producing a new row.
fn msb_fold_tablewise<F: Field>(table: &mut Vec<Vec<F>>, challenge: F) {
    if table.len() <= 1 {
        return;
    }
    let half = table.len() / 2;
    for k in 0..half {
        // Split to get non-overlapping mutable + shared references.
        let (lo_part, hi_part) = table.split_at(half);
        let new_row: Vec<F> = lo_part[k]
            .iter()
            .zip(&hi_part[k])
            .map(|(&lo, &hi)| lo + challenge * (hi - lo))
            .collect();
        table[k] = new_row;
    }
    table.truncate(half);
}

// ─── MSB evaluate helpers ──────────────────────────────────────────────────

/// MSB pairing: pair index `k` with `k + half` (not `2k` with `2k+1`).
fn msb_sequential_evaluate_into<F: Field>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    half: usize,
    coeffs_out: &mut [F],
) {
    for c in coeffs_out.iter_mut() {
        *c = F::ZERO;
    }
    let mut tw_buf: [(&[F], &[F]); 16] = [(&[], &[]); 16];
    let mut pw_buf: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];

    for k in 0..half {
        for (i, table) in tablewise.iter().enumerate() {
            tw_buf[i] = (&table[k], &table[k + half]);
        }
        for (i, table) in pairwise.iter().enumerate() {
            pw_buf[i] = (table[k], table[k + half]);
        }
        evaluator.accumulate_pair(coeffs_out, &tw_buf[..n_tw], &pw_buf[..n_pw]);
    }
}

#[cfg(feature = "parallel")]
fn msb_parallel_evaluate<F: Field>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    half: usize,
    n_coeffs: usize,
) -> Vec<F> {
    (0..half)
        .into_par_iter()
        .fold_with(vec![F::ZERO; n_coeffs], |mut acc, k| {
            let mut tw_buf: [(&[F], &[F]); 16] = [(&[], &[]); 16];
            let mut pw_buf: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];
            for (i, table) in tablewise.iter().enumerate() {
                tw_buf[i] = (&table[k], &table[k + half]);
            }
            for (i, table) in pairwise.iter().enumerate() {
                pw_buf[i] = (table[k], table[k + half]);
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
fn msb_parallel_evaluate<F: Field>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    half: usize,
    n_coeffs: usize,
) -> Vec<F> {
    let mut coeffs = vec![F::ZERO; n_coeffs];
    msb_sequential_evaluate_into(
        evaluator,
        tablewise,
        pairwise,
        n_tw,
        n_pw,
        half,
        &mut coeffs,
    );
    coeffs
}

// ─── Horner evaluation ─────────────────────────────────────────────────────

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

// ─── SumcheckProver impl ───────────────────────────────────────────────────

#[cfg(feature = "arkworks")]
impl<'a, F, E> SumcheckProver<F> for CoefficientProver<'a, F, E>
where
    F: ark_ff::Field,
    E: RoundPolyEvaluator<F>,
{
    fn degree(&self) -> usize {
        self.deg
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        if let Some(c) = challenge {
            self.reduce(c);
        }

        let coeffs = self.evaluate_coefficients();

        let mut evals = Vec::with_capacity(self.deg + 1);
        for i in 0..=self.deg {
            evals.push(eval_poly_at(&coeffs, F::from(i as u64)));
        }
        evals
    }

    fn finalize(&mut self, last_challenge: F) {
        self.reduce(last_challenge);
    }

    fn final_value(&self) -> F {
        // After full reduction, each pairwise table has 1 element and
        // each tablewise table has 1 row. The final value is the
        // evaluation of the user's polynomial at this single point.
        //
        // For degree-1 single-pairwise (multilinear shape): just pairwise[0][0].
        // General case: evaluate the round polynomial at the single "pair"
        // (which is really just one element — lo with implicit zero hi).
        if self.n_pw == 1 && self.n_tw == 0 && self.pairwise[0].len() == 1 {
            return self.pairwise[0][0];
        }

        // General: build the evaluator's output from the singleton tables.
        let mut coeffs = vec![F::ZERO; self.deg + 1];
        let mut tw_buf: [(&[F], &[F]); 16] = [(&[], &[]); 16];
        let mut pw_buf: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];
        for (i, table) in self.tablewise.iter().enumerate() {
            if table.len() == 1 {
                tw_buf[i] = (&table[0], &[]);
            }
        }
        for (i, table) in self.pairwise.iter().enumerate() {
            if table.len() == 1 {
                pw_buf[i] = (table[0], F::ZERO);
            }
        }
        self.evaluator
            .accumulate_pair(&mut coeffs, &tw_buf[..self.n_tw], &pw_buf[..self.n_pw]);
        eval_poly_at(&coeffs, F::ZERO) + eval_poly_at(&coeffs, F::ONE)
    }
}

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
            let (lo, hi) = pw[0];
            coeffs[0] += lo;
            coeffs[1] += hi - lo;
        }
    }

    /// MSB CoefficientProver produces valid round polynomials.
    #[test]
    fn msb_degree1_consistency() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        let evaluator = Degree1Eval;
        let mut prover = CoefficientProver::new(&evaluator, vec![], vec![evals]);
        let mut trng = StdRng::seed_from_u64(99);
        let mut t = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, 4, &mut t, |_, _| {});

        assert_eq!(
            proof.round_polys[0][0] + proof.round_polys[0][1],
            claimed_sum
        );

        let mut claim = claimed_sum;
        for (rp, &r) in proof.round_polys.iter().zip(&proof.challenges) {
            assert_eq!(rp[0] + rp[1], claim);
            claim = rp[0] + r * (rp[1] - rp[0]);
        }
        assert_eq!(proof.final_value, claim);
    }

    /// MSB CoefficientProver matches MSB MultilinearProver for degree 1.
    #[test]
    fn msb_matches_multilinear_prover() {
        use crate::provers::multilinear::MultilinearProver;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // MultilinearProver (MSB).
        let mut ml = MultilinearProver::new(evals.clone());
        let mut trng = StdRng::seed_from_u64(99);
        let mut t1 = SanityTranscript::new(&mut trng);
        let ml_proof = sumcheck(&mut ml, 4, &mut t1, |_, _| {});

        // CoefficientProver (MSB) with degree-1 evaluator.
        let evaluator = Degree1Eval;
        let mut cp = CoefficientProver::new(&evaluator, vec![], vec![evals]);
        let mut trng2 = StdRng::seed_from_u64(99);
        let mut t2 = SanityTranscript::new(&mut trng2);
        let cp_proof = sumcheck(&mut cp, 4, &mut t2, |_, _| {});

        // Same challenges (same transcript seed).
        assert_eq!(ml_proof.challenges, cp_proof.challenges);

        // Same round polynomial evaluations.
        for (i, (ml_rp, cp_rp)) in ml_proof
            .round_polys
            .iter()
            .zip(&cp_proof.round_polys)
            .enumerate()
        {
            assert_eq!(ml_rp[0], cp_rp[0], "round {i}: q(0) mismatch");
            assert_eq!(ml_rp[1], cp_rp[1], "round {i}: q(1) mismatch");
        }
    }
}
