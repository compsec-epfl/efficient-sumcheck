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
/// Wire format: evaluations `[q(0), q(1), q(2)]` where `q` is the degree-2
/// round polynomial.
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

        // Compute round polynomial evaluations at {0, 1, 2}.
        //
        // The round polynomial q(X) = Σ_{x'} f(r, X, x') · g(r, X, x').
        // With MSB half-split:
        //   f(r, X, x') = (1−X)·a_lo[x'] + X·a_hi[x']
        //
        // q(0) = dot(a_lo, b_lo)
        // q(1) = dot(a_hi, b_hi)
        // q(2) = dot(2·a_hi − a_lo, 2·b_hi − b_lo)
        let n = self.a.len();
        if n <= 1 {
            let v = if n == 1 {
                self.a[0] * self.b[0]
            } else {
                F::ZERO
            };
            return vec![v, F::ZERO, F::ZERO];
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
        let mut q1 = F::ZERO;
        let mut q2 = F::ZERO;
        for i in 0..paired {
            let al = a_lo_paired[i];
            let ah = a_hi[i];
            let bl = b_lo_paired[i];
            let bh = b_hi[i];
            q0 += al * bl;
            q1 += ah * bh;
            // f(2) = 2·ah − al, g(2) = 2·bh − bl
            let a2 = ah + ah - al;
            let b2 = bh + bh - bl;
            q2 += a2 * b2;
        }

        // Tail (hi is implicitly zero): contributes to q0 only.
        // q(0) += dot(tail_a, tail_b), q(1) += 0, q(2) += dot(-tail_a, -tail_b) = dot(tail_a, tail_b).
        let tail_dot: F = a_lo_tail.iter().zip(b_lo_tail).map(|(&a, &b)| a * b).sum();
        q0 += tail_dot;
        q2 += tail_dot;

        vec![q0, q1, q2]
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

        // Compare round-by-round consistency.
        assert_eq!(old_result.prover_messages.len(), new_result.round_polys.len());
        for (i, (old_msg, new_evals)) in old_result
            .prover_messages
            .iter()
            .zip(&new_result.round_polys)
            .enumerate()
        {
            let (c0, c2) = *old_msg;
            // Old API: c0 = q(0), and q(0) + q(1) = claim.
            // New API: [q(0), q(1), q(2)].
            assert_eq!(c0, new_evals[0], "round {i}: q(0) mismatch");
            // c2 from old = x² coefficient. Verify via:
            // q(X) = c0 + c1·X + c2·X²
            // q(2) = c0 + 2·c1 + 4·c2
            // c1 = q(1) - c0 - c2
            let q1 = new_evals[1];
            let c1_derived = q1 - c0 - c2;
            let q2_expected = c0 + c1_derived.double() + c2.double().double();
            assert_eq!(
                q2_expected, new_evals[2],
                "round {i}: q(2) inconsistent with (c0, c2)"
            );
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
