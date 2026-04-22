//! Inner-product sumcheck prover: `g = f_tilde * g_tilde`, degree 2.
//!
//! Implements [`SumcheckProver`] for the quadratic sumcheck `∑_x f(x)·g(x)`.

use crate::field::SumcheckField;
use crate::inner_product_sumcheck as ip;
use crate::sumcheck_prover::SumcheckProver;

/// Inner-product sumcheck prover (degree 2).
///
/// Computes `∑_x f(x)·g(x)` where `f` and `g` are multilinear polynomials
/// specified by their evaluations over the Boolean hypercube.
///
/// Wire format (EvalsInfty): `[q(0), q(∞)]` where `q` is the degree-2
/// round polynomial and `q(∞) = Σ (a_hi − a_lo)·(b_hi − b_lo)` is the
/// leading coefficient. Verifier derives `q(1) = claim - q(0)`.
///
/// # Construction
///
/// ```ignore
/// let mut prover = InnerProductProver::new(a, b);
/// let proof = sumcheck(&mut prover, num_rounds, &mut transcript, |_, _| {});
/// let (f_eval, g_eval) = prover.final_evaluations();
/// ```
pub struct InnerProductProver<F: SumcheckField> {
    a: Vec<F>,
    b: Vec<F>,
}

impl<F: SumcheckField> InnerProductProver<F> {
    /// Time strategy prover: holds both evaluation vectors in memory.
    pub fn new(a: Vec<F>, b: Vec<F>) -> Self {
        assert_eq!(a.len(), b.len(), "a and b must have equal length");
        Self { a, b }
    }

    /// Access the (possibly folded) evaluation vectors.
    pub fn evaluations(&self) -> (&[F], &[F]) {
        (&self.a, &self.b)
    }

    /// After full sumcheck: the final evaluations `(f(r), g(r))`.
    pub fn final_evaluations(&self) -> (F, F) {
        if self.a.len() == 1 {
            (self.a[0], self.b[0])
        } else {
            (F::ZERO, F::ZERO)
        }
    }
}

// NOTE: The `ark_ff::Field` bound is temporary — required because the
// underlying functions in `inner_product_sumcheck.rs` use `F: Field`.
// It will be removed when those functions are ported to `SumcheckField`.
#[cfg(feature = "arkworks")]
impl<F> SumcheckProver<F> for InnerProductProver<F>
where
    F: ark_ff::Field,
{
    fn degree(&self) -> usize {
        2
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        // Fold with previous challenge (if any).
        if let Some(w) = challenge {
            ip::fold(&mut self.a, w);
            ip::fold(&mut self.b, w);
        }

        // EvalsInfty: emit [q(0), q(∞)] where
        //   q(0)  = dot(a_lo, b_lo)
        //   q(∞)  = [x²] q(x) = dot(a_hi − a_lo, b_hi − b_lo)
        //         = Σ (ah − al)·(bh − bl)
        // The verifier derives q(1) = claim - q(0).
        let n = self.a.len();
        if n <= 1 {
            let v = if n == 1 {
                self.a[0] * self.b[0]
            } else {
                F::ZERO
            };
            return vec![v, F::ZERO];
        }

        let half = n.next_power_of_two() >> 1;
        let (a_lo, a_hi) = self.a.split_at(half);
        let (b_lo, b_hi) = self.b.split_at(half);

        // a_lo may be longer than a_hi (non-pow2 input, implicit zero padding).
        let paired = a_hi.len();
        let a_lo_paired = &a_lo[..paired];
        let b_lo_paired = &b_lo[..paired];
        let a_lo_tail = &a_lo[paired..];
        let b_lo_tail = &b_lo[paired..];

        let mut q0 = F::ZERO;
        let mut q_inf = F::ZERO;
        for i in 0..paired {
            let al = a_lo_paired[i];
            let ah = a_hi[i];
            let bl = b_lo_paired[i];
            let bh = b_hi[i];
            q0 += al * bl;
            q_inf += (ah - al) * (bh - bl);
        }

        // Tail (hi implicitly zero): ah = bh = 0, so
        //   q(0)  += al·bl
        //   q(∞)  += (0 − al)·(0 − bl) = al·bl
        let tail_dot: F = a_lo_tail.iter().zip(b_lo_tail).map(|(&a, &b)| a * b).sum();
        q0 += tail_dot;
        q_inf += tail_dot;

        vec![q0, q_inf]
    }

    fn finalize(&mut self, last_challenge: F) {
        ip::fold(&mut self.a, last_challenge);
        ip::fold(&mut self.b, last_challenge);
    }

    fn final_value(&self) -> F {
        if self.a.len() == 1 {
            self.a[0] * self.b[0]
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

    /// New `InnerProductProver` matches the old `inner_product_sumcheck`.
    ///
    /// The old API sends `(c0, c2)` (difference form); the new API sends
    /// `[q(0), q(1), q(2)]` (evaluation form). We verify the underlying
    /// values are consistent and the final evaluation matches.
    #[test]
    fn matches_legacy_inner_product_sumcheck() {
        let mut rng = StdRng::seed_from_u64(42);
        let a: Vec<F64> = (0..16).map(|_| F64::rand(&mut rng)).collect();
        let b: Vec<F64> = (0..16).map(|_| F64::rand(&mut rng)).collect();

        // Old API.
        let mut old_a = a.clone();
        let mut old_b = b.clone();
        let mut trng = StdRng::seed_from_u64(99);
        let mut t_old = SanityTranscript::new(&mut trng);
        let old_result = crate::inner_product_sumcheck::inner_product_sumcheck(
            &mut old_a,
            &mut old_b,
            &mut t_old,
            |_, _| {},
        );

        // New API.
        let mut prover = InnerProductProver::new(a, b);
        let num_rounds = 4; // log2(16)
        let mut trng2 = StdRng::seed_from_u64(99);
        let mut t_new = SanityTranscript::new(&mut trng2);
        let new_result = sumcheck(&mut prover, num_rounds, &mut t_new, |_, _| {});

        // Both APIs now emit the same wire format: (c0, c2) in difference
        // form (= EvalsInfty for degree 2 where c2 is the x² coefficient).
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
            let (c0, c2) = *old_msg;
            assert_eq!(new_evals.len(), 2, "round {i}: EvalsInfty degree-2 wire length");
            assert_eq!(c0, new_evals[0], "round {i}: q(0) mismatch");
            assert_eq!(c2, new_evals[1], "round {i}: q(∞) mismatch");
        }

        // Compare challenges (should be identical since same transcript seed).
        assert_eq!(old_result.verifier_messages, new_result.challenges);

        // Compare final evaluations.
        let (fa, fb) = prover.final_evaluations();
        let (old_fa, old_fb) = old_result.final_evaluations;
        assert_eq!(fa, old_fa, "f(r) mismatch");
        assert_eq!(fb, old_fb, "g(r) mismatch");

        // Final value = f(r) * g(r).
        assert_eq!(new_result.final_value, fa * fb);
    }
}
