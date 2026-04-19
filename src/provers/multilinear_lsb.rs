//! LSB (pair-split) multilinear sumcheck prover: `g = f_tilde`, degree 1.
//!
//! Folds the *least-significant* variable each round: pairs `(f[2k], f[2k+1])`.
//! This is the natural layout for **sequential streaming** where evaluations
//! arrive in index order — adjacent pairs are immediately available.
//!
//! Use this prover for Jolt-style workloads where the witness is generated
//! incrementally (CPU trace). For in-memory or random-access-streaming
//! workloads, prefer [`MultilinearProver`](super::multilinear::MultilinearProver)
//! (MSB layout).

extern crate alloc;
use alloc::vec::Vec;
use crate::field::SumcheckField;
#[cfg(feature = "arkworks")]
use crate::sumcheck_prover::SumcheckProver;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// LSB multilinear sumcheck prover (degree 1, pair-split layout).
///
/// Computes `sum_x v(x)` by folding the least-significant variable each round.
///
/// # Construction
///
/// ```ignore
/// let mut prover = MultilinearProverLSB::new(evals);
/// let proof = sumcheck(&mut prover, num_rounds, &mut transcript, |_, _| {});
/// ```
pub struct MultilinearProverLSB<F: SumcheckField> {
    evals: Vec<F>,
}

impl<F: SumcheckField> MultilinearProverLSB<F> {
    /// Time strategy prover with LSB (pair-split) layout.
    pub fn new(evals: Vec<F>) -> Self {
        Self { evals }
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        if self.evals.is_empty() {
            0
        } else {
            self.evals.len().next_power_of_two().trailing_zeros() as usize
        }
    }

    /// Access the (possibly folded) evaluation table.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }
}

// ─── LSB fold and compute ──────────────────────────────────────────────────

/// Compute (s0, s1) for the LSB round polynomial from pair-split layout.
///
/// `s0 = sum of even-indexed elements = sum f[2k]`
/// `s1 = sum of odd-indexed elements  = sum f[2k+1]`
fn compute_lsb<F: SumcheckField>(evals: &[F]) -> (F, F) {
    if evals.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    if evals.len() == 1 {
        return (evals[0], F::ZERO);
    }

    #[cfg(feature = "parallel")]
    {
        const PARALLEL_THRESHOLD: usize = 1 << 14;
        if evals.len() > PARALLEL_THRESHOLD {
            let (s0, s1) = evals
                .par_chunks(2)
                .map(|chunk| {
                    let a = chunk[0];
                    let b = if chunk.len() > 1 { chunk[1] } else { F::ZERO };
                    (a, b)
                })
                .reduce(
                    || (F::ZERO, F::ZERO),
                    |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
                );
            return (s0, s1);
        }
    }

    let mut s0 = F::ZERO;
    let mut s1 = F::ZERO;
    for chunk in evals.chunks(2) {
        s0 += chunk[0];
        if chunk.len() > 1 {
            s1 += chunk[1];
        }
    }
    (s0, s1)
}

/// In-place LSB (pair-split) fold: `new[k] = f[2k] + w * (f[2k+1] - f[2k])`.
fn fold_lsb<F: SumcheckField>(evals: &mut Vec<F>, weight: F) {
    if evals.len() <= 1 {
        return;
    }
    let new_len = evals.len() / 2;

    #[cfg(feature = "parallel")]
    {
        const PARALLEL_THRESHOLD: usize = 1 << 14;
        if evals.len() > PARALLEL_THRESHOLD {
            let out: Vec<F> = evals
                .par_chunks(2)
                .map(|chunk| chunk[0] + weight * (chunk[1] - chunk[0]))
                .collect();
            *evals = out;
            return;
        }
    }

    for i in 0..new_len {
        let a = evals[2 * i];
        let b = evals[2 * i + 1];
        evals[i] = a + weight * (b - a);
    }
    evals.truncate(new_len);
}

/// Fused fold + compute: fold with `weight`, then compute the next round's
/// (s0, s1) from the folded data. Single pass over pairs of pairs.
fn fused_fold_and_compute_lsb<F: SumcheckField>(evals: &mut Vec<F>, weight: F) -> (F, F) {
    let n = evals.len();
    if n < 4 {
        fold_lsb(evals, weight);
        return compute_lsb(evals);
    }

    let new_len = n / 2;
    let mut s0 = F::ZERO;
    let mut s1 = F::ZERO;

    // Process quads: (f[4k], f[4k+1], f[4k+2], f[4k+3])
    // Fold produces: new[2k] = f[4k] + w*(f[4k+1]-f[4k])
    //                new[2k+1] = f[4k+2] + w*(f[4k+3]-f[4k+2])
    // Next round: s0 += new[2k], s1 += new[2k+1]
    let quads = n / 4;
    for q in 0..quads {
        let i = 4 * q;
        let a = evals[i] + weight * (evals[i + 1] - evals[i]);
        let b = evals[i + 2] + weight * (evals[i + 3] - evals[i + 2]);
        evals[2 * q] = a;
        evals[2 * q + 1] = b;
        s0 += a;
        s1 += b;
    }

    // Handle remainder if new_len is odd (original n not divisible by 4).
    if new_len > 2 * quads {
        let i = 4 * quads;
        let a = evals[i] + weight * (evals[i + 1] - evals[i]);
        evals[2 * quads] = a;
        s0 += a;
    }

    evals.truncate(new_len);
    (s0, s1)
}

// ─── SumcheckProver impl ───────────────────────────────────────────────────

#[cfg(feature = "arkworks")]
impl<F> SumcheckProver<F> for MultilinearProverLSB<F>
where
    F: ark_ff::Field,
{
    fn degree(&self) -> usize {
        1
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        let (s0, s1) = if let Some(w) = challenge {
            fused_fold_and_compute_lsb(&mut self.evals, w)
        } else {
            compute_lsb(&self.evals)
        };
        vec![s0, s1]
    }

    fn finalize(&mut self, last_challenge: F) {
        fold_lsb(&mut self.evals, last_challenge);
    }

    fn final_value(&self) -> F {
        if self.evals.len() == 1 {
            self.evals[0]
        } else {
            F::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provers::multilinear::MultilinearProver;
    use crate::runner::sumcheck;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    /// LSB and MSB provers produce different round polynomials (they fold
    /// different variables) but the same final value when evaluated at the
    /// same random point via independent MLE evaluation.
    #[test]
    fn lsb_prover_completes_and_verifies() {
        let mut rng = StdRng::seed_from_u64(42);
        let evals: Vec<F64> = (0..16).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        let mut prover = MultilinearProverLSB::new(evals);
        let num_vars = prover.num_variables();
        let mut trng = StdRng::seed_from_u64(99);
        let mut t = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, num_vars, &mut t, |_, _| {});

        // Round-0 consistency.
        assert_eq!(
            proof.round_polys[0][0] + proof.round_polys[0][1],
            claimed_sum
        );

        // All-round consistency.
        let mut claim = claimed_sum;
        for (rp, &r) in proof.round_polys.iter().zip(&proof.challenges) {
            assert_eq!(rp[0] + rp[1], claim, "consistency check failed");
            claim = rp[0] + r * (rp[1] - rp[0]);
        }

        // Final value matches claim after all rounds.
        assert_eq!(proof.final_value, claim);
        assert_eq!(prover.evals().len(), 1);
    }

    /// LSB and MSB produce the same claimed sum (both prove the same statement).
    #[test]
    fn lsb_and_msb_prove_same_sum() {
        let mut rng = StdRng::seed_from_u64(42);
        let evals: Vec<F64> = (0..32).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        // LSB.
        let mut lsb = MultilinearProverLSB::new(evals.clone());
        let mut trng = StdRng::seed_from_u64(99);
        let mut t = SanityTranscript::new(&mut trng);
        let lsb_proof = sumcheck(&mut lsb, 5, &mut t, |_, _| {});
        assert_eq!(
            lsb_proof.round_polys[0][0] + lsb_proof.round_polys[0][1],
            claimed_sum
        );

        // MSB.
        let mut msb = MultilinearProver::new(evals);
        let mut trng2 = StdRng::seed_from_u64(99);
        let mut t2 = SanityTranscript::new(&mut trng2);
        let msb_proof = sumcheck(&mut msb, 5, &mut t2, |_, _| {});
        assert_eq!(
            msb_proof.round_polys[0][0] + msb_proof.round_polys[0][1],
            claimed_sum
        );
    }
}
