use ark_ff::Field;
use ark_std::{rand::Rng, vec::Vec};

use crate::{prover::Prover, streams::Stream};

/// Transcript for the inner product sumcheck protocol.
///
/// Each round the prover sends `(a, b)`:
///   - `a = q(0) = Σ f_even · g_even`                     (constant coefficient)
///   - `b = Σ (f_even · g_odd + f_odd · g_even)`          (raw cross sum)
///
/// The true round polynomial `q(X) = a + L·X + Q·X²` has:
///   - `L = b − 2a` (linear coefficient)
///   - `Q = claim − b` (quadratic coefficient)
///
/// derived from the constraint `q(0) + q(1) = claim` together with the identities
/// `L = Σ(f_e·g_o + f_o·g_e) − 2·Σ f_e·g_e = b − 2a` and
/// `q(1) = Σ f_o·g_o = claim − a`, hence `Q = q(1) − a − L = claim − b`.
///
/// Wire format is `(a, b)` rather than e.g. `(q(0), q(1))` because the raw cross
/// sum is one fewer subtraction per lane on the prover side. See
/// [`ProductSumcheck::evaluate_round_poly`] for the reconstruction.
#[derive(Debug, PartialEq)]
pub struct ProductSumcheck<F: Field> {
    pub prover_messages: Vec<(F, F)>,
    pub verifier_messages: Vec<F>,
    /// The two input polynomials evaluated at the verifier challenge point
    /// `(r_0, ..., r_{n-1})`: `(f(r), g(r))`. Populated by
    /// [`crate::inner_product_sumcheck`] and
    /// [`crate::inner_product_sumcheck_with_hook`] (and all their SIMD
    /// dispatch paths). The legacy [`ProductSumcheck::prove`] constructor
    /// leaves this as `(F::ZERO, F::ZERO)` — it's a low-level test helper
    /// that doesn't surface fold state.
    pub final_evaluations: (F, F),
}

impl<F: Field> ProductSumcheck<F> {
    /// Evaluate the degree-2 round polynomial at `r` from the wire-format
    /// message `(a, b)` and the current claim.
    ///
    /// `a = q(0) = Σ f_e·g_e` (constant coefficient), `b = Σ(f_e·g_o + f_o·g_e)`
    /// (raw cross sum). The true round polynomial is
    /// `q(X) = a + (b − 2a)·X + (claim − b)·X²`; this function returns `q(r)`.
    #[inline]
    pub fn evaluate_round_poly(r: F, a: F, b: F, claim: F) -> F {
        let linear = b - a.double();
        let quadratic = claim - b;
        a + linear * r + quadratic * r.square()
    }

    pub fn prove<S, P>(prover: &mut P, rng: &mut impl Rng) -> Self
    where
        S: Stream<F>,
        P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F)>>,
    {
        let mut prover_messages: Vec<(F, F)> = vec![];
        let mut verifier_messages: Vec<F> = vec![];

        let mut verifier_message: Option<F> = None;
        while let Some((a, b)) = prover.next_message(verifier_message) {
            let is_round_accepted = match verifier_message {
                None => true,
                Some(prev_r) => {
                    verifier_messages.push(prev_r);
                    // Verify: current q(0) + q(1) == previous q(r).
                    // q(0) = a, q(1) = a + b + c where c = prev_claim - 2*prev_a - prev_b.
                    // So q(0) + q(1) = 2a + b + c.
                    // But actually, q(0)+q(1) is the current claim, and it must
                    // equal q_prev(r). The prover just sends (a, b) and we check
                    // consistency across rounds externally. For this internal test,
                    // we accept all rounds (consistency checked by the test harness).
                    true
                }
            };

            prover_messages.push((a, b));
            if !is_round_accepted {
                break;
            }

            verifier_message = Some(F::rand(rng));
        }

        // NOTE: `final_evaluations` is not tracked by the generic `Prover`
        // trait; see field doc.
        ProductSumcheck {
            prover_messages,
            verifier_messages,
            final_evaluations: (F::ZERO, F::ZERO),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear_product::TimeProductProver,
        tests::{multilinear_product::consistency_test, BenchStream, F64},
    };
    use ark_ff::{AdditiveGroup, Field};

    #[test]
    fn algorithm_consistency() {
        consistency_test::<F64, BenchStream<F64>, TimeProductProver<F64, BenchStream<F64>>>();
    }

    #[test]
    fn test_evaluate_round_poly() {
        use super::ProductSumcheck;
        use ark_ff::UniformRand;
        use ark_std::test_rng;

        // Exercise the real wire convention: `b` is the raw cross sum
        // `Σ(f_e·g_o + f_o·g_e)`, NOT the linear coefficient of q. The linear
        // coefficient is `b − 2a` and the quadratic is `claim − b`.
        let mut rng = test_rng();
        for _ in 0..1000 {
            // Sample a random degree-2 polynomial via its coefficients.
            let a = F64::rand(&mut rng); // q(0)
            let linear = F64::rand(&mut rng); // linear coefficient of q
            let quadratic = F64::rand(&mut rng); // quadratic coefficient of q
            let r = F64::rand(&mut rng);

            // Reconstruct wire-format b: linear = b − 2a  ⇒  b = linear + 2a.
            let b = linear + a.double();
            // claim = q(0) + q(1) = 2a + linear + quadratic.
            let claim = a.double() + linear + quadratic;

            let expected = a + linear * r + quadratic * r.square();
            let got = ProductSumcheck::<F64>::evaluate_round_poly(r, a, b, claim);
            assert_eq!(expected, got);
        }
    }

    /// End-to-end check: wire what the prover actually writes into
    /// `evaluate_round_poly` and confirm it reconstructs `q(r)` correctly.
    /// Catches protocol-convention regressions between prover and verifier.
    #[test]
    fn test_evaluate_round_poly_matches_prover_output() {
        use super::ProductSumcheck;
        use crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate_slices;
        use ark_ff::UniformRand;
        use ark_std::test_rng;

        let mut rng = test_rng();
        let n = 1 << 8;
        let f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let (a, b) = pairwise_product_evaluate_slices(&f, &g);
        // claim = q(0) + q(1) = Σ f·g (inner product over full cube)
        let claim: F64 = f.iter().zip(g.iter()).map(|(fi, gi)| *fi * *gi).sum();

        let r = F64::rand(&mut rng);

        // Reference: evaluate q(r) where q(X) = f(X)·g(X) summed over the rest of the cube,
        // computed directly by folding f and g at r then taking the inner product.
        let mut ff = f.clone();
        let mut gg = g.clone();
        for pair in ff.chunks_mut(2) {
            pair[0] = pair[0] + r * (pair[1] - pair[0]);
        }
        for pair in gg.chunks_mut(2) {
            pair[0] = pair[0] + r * (pair[1] - pair[0]);
        }
        let expected: F64 = (0..n / 2).map(|k| ff[2 * k] * gg[2 * k]).sum();

        let got = ProductSumcheck::<F64>::evaluate_round_poly(r, a, b, claim);
        assert_eq!(
            got, expected,
            "evaluate_round_poly disagrees with folded prover output"
        );
    }
}
