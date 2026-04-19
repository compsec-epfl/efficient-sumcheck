//! Multilinear sumcheck prover: `g = f_tilde`, degree 1.
//!
//! Wraps the fused fold+compute kernel from `multilinear_sumcheck.rs`
//! behind the [`SumcheckProver`] trait.

use crate::field::SumcheckField;
use crate::multilinear_sumcheck::{
    compute_sumcheck_polynomial, fold, fused_fold_and_compute_polynomial,
};
use crate::sumcheck_prover::SumcheckProver;

/// Multilinear sumcheck prover (degree 1).
///
/// Computes `∑_x v(x)` where `v` is a multilinear polynomial specified
/// by its evaluations over the Boolean hypercube.
///
/// # Construction
///
/// ```ignore
/// // Time strategy: O(2^v) space, O(2^v) total time.
/// let mut prover = MultilinearProver::new(evals);
/// let proof = sumcheck(&mut prover, num_rounds, &mut transcript, |_, _| {});
/// ```
pub struct MultilinearProver<F: SumcheckField> {
    evals: Vec<F>,
}

impl<F: SumcheckField> MultilinearProver<F> {
    /// Time strategy prover: holds all evaluations in memory.
    pub fn new(evals: Vec<F>) -> Self {
        Self { evals }
    }

    /// Number of variables (log2 of evaluation count, rounded up).
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

// NOTE: The `ark_ff::Field` bound is temporary — required because the
// underlying functions in `multilinear_sumcheck.rs` use `F: Field`.
// It will be removed when those functions are ported to `SumcheckField`.
#[cfg(feature = "arkworks")]
impl<F> SumcheckProver<F> for MultilinearProver<F>
where
    F: ark_ff::Field,
{
    fn degree(&self) -> usize {
        1
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        let (s0, s1) = if let Some(w) = challenge {
            fused_fold_and_compute_polynomial(&mut self.evals, w)
        } else {
            compute_sumcheck_polynomial(&self.evals)
        };
        vec![s0, s1]
    }

    fn finalize(&mut self, last_challenge: F) {
        fold(&mut self.evals, last_challenge);
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
    use crate::runner::sumcheck;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    /// New `MultilinearProver` API produces the same proof as the old
    /// `multilinear_sumcheck` function.
    #[test]
    fn matches_legacy_multilinear_sumcheck() {
        use ark_ff::Field;
        let mut rng = StdRng::seed_from_u64(42);
        let evals: Vec<F64> = (0..16).map(|_| F64::rand(&mut rng)).collect();

        // Old API.
        let mut old_evals = evals.clone();
        let mut trng = StdRng::seed_from_u64(99);
        let mut t_old = SanityTranscript::new(&mut trng);
        let old_result = crate::multilinear_sumcheck::multilinear_sumcheck(
            &mut old_evals,
            &mut t_old,
            |_, _| {},
        );

        // New API.
        let mut prover = MultilinearProver::new(evals);
        let num_rounds = prover.num_variables();
        let mut trng2 = StdRng::seed_from_u64(99);
        let mut t_new = SanityTranscript::new(&mut trng2);
        let new_result = sumcheck(&mut prover, num_rounds, &mut t_new, |_, _| {});

        // Compare round polynomials.
        assert_eq!(
            old_result.prover_messages.len(),
            new_result.round_polys.len()
        );
        for (i, (old_msg, new_evals)) in old_result
            .prover_messages
            .iter()
            .zip(&new_result.round_polys)
            .enumerate()
        {
            assert_eq!(old_msg.0, new_evals[0], "round {i}: s0 mismatch");
            assert_eq!(old_msg.1, new_evals[1], "round {i}: s1 mismatch");
        }

        // Compare challenges.
        assert_eq!(old_result.verifier_messages, new_result.challenges);

        // Compare final value.
        assert_eq!(old_result.final_evaluation, new_result.final_value);
    }
}
