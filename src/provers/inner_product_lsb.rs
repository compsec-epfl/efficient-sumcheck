//! LSB (pair-split) inner-product sumcheck prover: `g = f_tilde * g_tilde`, degree 2.
//!
//! Folds the *least-significant* variable each round: pairs `(f[2k], f[2k+1])`.
//! This is the natural layout for **sequential streaming** where evaluations
//! arrive in index order.
//!
//! Use this prover for Jolt-style workloads. For in-memory or random-access
//! workloads, prefer [`InnerProductProver`](super::inner_product::InnerProductProver)
//! (MSB layout).

extern crate alloc;
use crate::field::SumcheckField;
#[cfg(feature = "arkworks")]
use crate::sumcheck_prover::SumcheckProver;
use alloc::vec::Vec;

/// LSB inner-product sumcheck prover (degree 2, pair-split layout).
///
/// Computes `sum_x f(x) * g(x)` by folding the least-significant variable
/// each round.
///
/// ```ignore
/// let mut prover = InnerProductProverLSB::new(a, b);
/// let proof = sumcheck(&mut prover, num_rounds, &mut transcript, |_, _| {});
/// let (f_r, g_r) = prover.final_evaluations();
/// ```
pub struct InnerProductProverLSB<F: SumcheckField> {
    a: Vec<F>,
    b: Vec<F>,
}

impl<F: SumcheckField> InnerProductProverLSB<F> {
    pub fn new(a: Vec<F>, b: Vec<F>) -> Self {
        assert_eq!(a.len(), b.len(), "a and b must have equal length");
        Self { a, b }
    }

    pub fn evaluations(&self) -> (&[F], &[F]) {
        (&self.a, &self.b)
    }

    pub fn final_evaluations(&self) -> (F, F) {
        if self.a.len() == 1 {
            (self.a[0], self.b[0])
        } else {
            (F::ZERO, F::ZERO)
        }
    }
}

// ─── LSB fold and compute ──────────────────────────────────────────────────

/// Compute EvalsInfty round polynomial `(q(0), q(∞))` from LSB pair-split layout.
///
/// q(0)  = sum a[2k] * b[2k]
/// q(∞)  = [x²] q(x) = sum (a[2k+1] - a[2k]) * (b[2k+1] - b[2k])
fn compute_lsb<F: SumcheckField>(a: &[F], b: &[F]) -> (F, F) {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    if a.len() == 1 {
        return (a[0] * b[0], F::ZERO);
    }

    let mut q0 = F::ZERO;
    let mut q_inf = F::ZERO;
    for i in (0..a.len()).step_by(2) {
        let a_even = a[i];
        let a_odd = a[i + 1];
        let b_even = b[i];
        let b_odd = b[i + 1];
        q0 += a_even * b_even;
        q_inf += (a_odd - a_even) * (b_odd - b_even);
    }
    (q0, q_inf)
}

/// In-place LSB fold: `new[k] = f[2k] + w * (f[2k+1] - f[2k])`.
fn fold_lsb<F: SumcheckField>(v: &mut Vec<F>, weight: F) {
    if v.len() <= 1 {
        return;
    }
    let new_len = v.len() / 2;
    for i in 0..new_len {
        let a = v[2 * i];
        let b = v[2 * i + 1];
        v[i] = a + weight * (b - a);
    }
    v.truncate(new_len);
}

/// Fused fold + compute: fold both vectors with `weight`, then compute
/// the next round's EvalsInfty `(q(0), q(∞))` in one pass over quads.
fn fused_fold_and_compute_lsb<F: SumcheckField>(
    a: &mut Vec<F>,
    b: &mut Vec<F>,
    weight: F,
) -> (F, F) {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    if n < 4 {
        fold_lsb(a, weight);
        fold_lsb(b, weight);
        return compute_lsb(a, b);
    }

    let new_len = n / 2;
    let mut q0 = F::ZERO;
    let mut q_inf = F::ZERO;

    // Process quads: indices (4k, 4k+1, 4k+2, 4k+3)
    // Fold produces: new_a[2k]   = a[4k]   + w*(a[4k+1] - a[4k])
    //                new_a[2k+1] = a[4k+2] + w*(a[4k+3] - a[4k+2])
    // Next round's LSB pairs are (new[2k], new[2k+1]).
    let quads = n / 4;
    for q in 0..quads {
        let i = 4 * q;
        let na_even = a[i] + weight * (a[i + 1] - a[i]);
        let na_odd = a[i + 2] + weight * (a[i + 3] - a[i + 2]);
        let nb_even = b[i] + weight * (b[i + 1] - b[i]);
        let nb_odd = b[i + 2] + weight * (b[i + 3] - b[i + 2]);

        a[2 * q] = na_even;
        a[2 * q + 1] = na_odd;
        b[2 * q] = nb_even;
        b[2 * q + 1] = nb_odd;

        q0 += na_even * nb_even;
        q_inf += (na_odd - na_even) * (nb_odd - nb_even);
    }

    // Handle remainder if new_len is odd.
    if new_len > 2 * quads {
        let i = 4 * quads;
        let na = a[i] + weight * (a[i + 1] - a[i]);
        let nb = b[i] + weight * (b[i + 1] - b[i]);
        a[2 * quads] = na;
        b[2 * quads] = nb;
        q0 += na * nb;
    }

    a.truncate(new_len);
    b.truncate(new_len);
    (q0, q_inf)
}

// ─── SumcheckProver impl ───────────────────────────────────────────────────

#[cfg(feature = "arkworks")]
impl<F> SumcheckProver<F> for InnerProductProverLSB<F>
where
    F: ark_ff::Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
{
    fn degree(&self) -> usize {
        2
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        // EvalsInfty: degree 2 → emit [q(0), q(∞)].
        let (q0, q_inf) = if let Some(w) = challenge {
            fused_fold_and_compute_lsb(&mut self.a, &mut self.b, w)
        } else {
            compute_lsb(&self.a, &self.b)
        };
        vec![q0, q_inf]
    }

    fn finalize(&mut self, last_challenge: F) {
        fold_lsb(&mut self.a, last_challenge);
        fold_lsb(&mut self.b, last_challenge);
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
    use crate::provers::inner_product::InnerProductProver;
    use crate::runner::sumcheck;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn lsb_inner_product_completes_and_verifies() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 16;
        let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();

        let mut prover = InnerProductProverLSB::new(a.clone(), b.clone());
        let num_vars = 4;
        let mut trng = StdRng::seed_from_u64(99);
        let mut t = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, num_vars, &mut t, |_, _| {});

        // EvalsInfty: degree-2 wire is [q(0), q(∞)]. Derive q(1) = claim - q(0),
        // then reconstruct q(r) = q(0) + r·(q(1) - q(0) - q(∞)) + q(∞)·r².
        let mut claim = claimed_sum;
        for (rp, &r) in proof.round_polys.iter().zip(&proof.challenges) {
            assert_eq!(rp.len(), 2, "EvalsInfty degree-2 wire length");
            let q0 = rp[0];
            let q_inf = rp[1];
            let q1 = claim - q0;
            claim = q0 + r * (q1 - q0 - q_inf) + q_inf * r * r;
        }
        assert_eq!(proof.final_value, claim);

        // Final evaluations.
        let (fa, fb) = prover.final_evaluations();
        assert_eq!(proof.final_value, fa * fb);
    }

    #[test]
    fn lsb_and_msb_prove_same_sum() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 32;
        let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();

        // Both provers emit EvalsInfty [q(0), q(∞)]; verify by reconstructing
        // q(1) = claim - q(0) and checking the sum equals the claimed sum.
        let mut lsb = InnerProductProverLSB::new(a.clone(), b.clone());
        let mut trng = StdRng::seed_from_u64(99);
        let mut t = SanityTranscript::new(&mut trng);
        let lsb_proof = sumcheck(&mut lsb, 5, &mut t, |_, _| {});
        let lsb_q1 = claimed_sum - lsb_proof.round_polys[0][0];
        assert_eq!(lsb_proof.round_polys[0][0] + lsb_q1, claimed_sum);

        let mut msb = InnerProductProver::new(a, b);
        let mut trng2 = StdRng::seed_from_u64(99);
        let mut t2 = SanityTranscript::new(&mut trng2);
        let msb_proof = sumcheck(&mut msb, 5, &mut t2, |_, _| {});
        let msb_q1 = claimed_sum - msb_proof.round_polys[0][0];
        assert_eq!(msb_proof.round_polys[0][0] + msb_q1, claimed_sum);
    }
}
