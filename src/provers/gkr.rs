//! GKR round sumcheck prover (degree 2).
//!
//! Implements [`SumcheckProver`] for the GKR round polynomial:
//!
//! ```text
//! f_r(b, c) = add_i(r, b, c) · (W(b) + W(c)) + mult_i(r, b, c) · (W(b) · W(c))
//! ```
//!
//! where `add_i` and `mult_i` are gate predicates (partially evaluated at the
//! previous layer's random point `r`), and `W` is the witness for the next
//! layer.
//!
//! # Construction
//!
//! The prover takes three evaluation tables:
//! - `add_evals`: `add_i(r, b, c)` over `{0,1}^{2k}`, `2^{2k}` entries
//! - `mult_evals`: `mult_i(r, b, c)` over `{0,1}^{2k}`, `2^{2k}` entries
//! - `w_evals`: `W(x)` over `{0,1}^k`, `2^k` entries
//!
//! The sumcheck runs over `2k` variables. After all rounds and finalization,
//! [`claimed_w_values()`](GkrProver::claimed_w_values) returns `(W(b*), W(c*))`
//! for the reduce-to-one sub-protocol.
//!
//! # Example
//!
//! ```ignore
//! let mut prover = GkrProver::new(add_evals, mult_evals, w_evals);
//! let proof = sumcheck(&mut prover, 2 * k, &mut transcript, noop_hook);
//! let (w_b, w_c) = prover.claimed_w_values();
//! ```

use crate::field::SumcheckField;
use crate::inner_product_sumcheck as ip;
use crate::sumcheck_prover::SumcheckProver;

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// GKR round sumcheck prover (degree 2).
///
/// See [module docs](self) for details.
pub struct GkrProver<F: SumcheckField> {
    /// Gate add predicate: `add_i(r, b, c)`, `2^{2k}` entries.
    add_evals: Vec<F>,
    /// Gate mult predicate: `mult_i(r, b, c)`, `2^{2k}` entries.
    mult_evals: Vec<F>,
    /// Witness `W(b)` broadcast over c: `w_b[b * 2^k + c] = W(b)`.
    w_b: Vec<F>,
    /// Witness `W(c)` broadcast over b: `w_c[b * 2^k + c] = W(c)`.
    w_c: Vec<F>,
}

impl<F: SumcheckField> GkrProver<F> {
    /// Construct from gate predicates and witness evaluations.
    ///
    /// - `add_evals`: `add_i(r, b, c)` for all `(b, c) in {0,1}^{2k}`.
    /// - `mult_evals`: `mult_i(r, b, c)` for all `(b, c) in {0,1}^{2k}`.
    /// - `w_evals`: `W(x)` for all `x in {0,1}^k`.
    ///
    /// The gate tables must have length `w_evals.len()^2`.
    pub fn new(add_evals: Vec<F>, mult_evals: Vec<F>, w_evals: Vec<F>) -> Self {
        let n = w_evals.len();
        let n_bc = n * n;
        assert_eq!(add_evals.len(), n_bc, "add_evals must have len w^2");
        assert_eq!(mult_evals.len(), n_bc, "mult_evals must have len w^2");

        // Expand witness into broadcast tables over the (b, c) hypercube.
        let mut w_b = vec![F::ZERO; n_bc];
        let mut w_c = vec![F::ZERO; n_bc];
        for b in 0..n {
            for c in 0..n {
                let idx = b * n + c;
                w_b[idx] = w_evals[b];
                w_c[idx] = w_evals[c];
            }
        }

        Self {
            add_evals,
            mult_evals,
            w_b,
            w_c,
        }
    }

    /// After full sumcheck: the claimed witness evaluations `(W(b*), W(c*))`.
    ///
    /// These are the inputs to the reduce-to-one sub-protocol (Thaler §4.5.2).
    pub fn claimed_w_values(&self) -> (F, F) {
        if self.w_b.len() == 1 {
            (self.w_b[0], self.w_c[0])
        } else {
            (F::ZERO, F::ZERO)
        }
    }
}

#[cfg(feature = "arkworks")]
impl<F> SumcheckProver<F> for GkrProver<F>
where
    F: ark_ff::Field,
{
    fn degree(&self) -> usize {
        2
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        // Fold all four tables with the previous challenge.
        if let Some(w) = challenge {
            ip::fold(&mut self.add_evals, w);
            ip::fold(&mut self.mult_evals, w);
            ip::fold(&mut self.w_b, w);
            ip::fold(&mut self.w_c, w);
        }

        let n = self.add_evals.len();
        if n <= 1 {
            let v = if n == 1 {
                let wb = self.w_b[0];
                let wc = self.w_c[0];
                self.add_evals[0] * (wb + wc) + self.mult_evals[0] * (wb * wc)
            } else {
                F::ZERO
            };
            return vec![v, F::ZERO, F::ZERO];
        }

        let half = n.next_power_of_two() >> 1;
        let (add_lo, add_hi) = self.add_evals.split_at(half);
        let (mult_lo, mult_hi) = self.mult_evals.split_at(half);
        let (wb_lo, wb_hi) = self.w_b.split_at(half);
        let (wc_lo, wc_hi) = self.w_c.split_at(half);

        let paired = add_hi.len();

        let mut q0 = F::ZERO;
        let mut q1 = F::ZERO;
        let mut q2 = F::ZERO;

        for i in 0..paired {
            let al = add_lo[i];
            let ah = add_hi[i];
            let ml = mult_lo[i];
            let mh = mult_hi[i];
            let wbl = wb_lo[i];
            let wbh = wb_hi[i];
            let wcl = wc_lo[i];
            let wch = wc_hi[i];

            // q(0): all factors at t=0 (low half)
            q0 += al * (wbl + wcl) + ml * (wbl * wcl);

            // q(1): all factors at t=1 (high half)
            q1 += ah * (wbh + wch) + mh * (wbh * wch);

            // q(2): linear extension to t=2: val_2 = 2*hi - lo
            let a2 = ah + ah - al;
            let m2 = mh + mh - ml;
            let wb2 = wbh + wbh - wbl;
            let wc2 = wch + wch - wcl;
            q2 += a2 * (wb2 + wc2) + m2 * (wb2 * wc2);
        }

        // Tail: hi is implicitly zero, so at t=2 each factor is -lo.
        // add term: (-al)*(-wbl + -wcl) = al*(wbl + wcl)  [even number of negations]
        // mult term: (-ml)*(-wbl)*(-wcl) = -ml*wbl*wcl    [odd number of negations]
        for i in paired..half.min(n) {
            let al = add_lo[i];
            let ml = mult_lo[i];
            let wbl = wb_lo[i];
            let wcl = wc_lo[i];

            q0 += al * (wbl + wcl) + ml * (wbl * wcl);
            // q(1) += 0 (hi is zero)
            q2 += al * (wbl + wcl) - ml * (wbl * wcl);
        }

        vec![q0, q1, q2]
    }

    fn finalize(&mut self, last_challenge: F) {
        ip::fold(&mut self.add_evals, last_challenge);
        ip::fold(&mut self.mult_evals, last_challenge);
        ip::fold(&mut self.w_b, last_challenge);
        ip::fold(&mut self.w_c, last_challenge);
    }

    fn final_value(&self) -> F {
        if self.add_evals.len() == 1 {
            let wb = self.w_b[0];
            let wc = self.w_c[0];
            self.add_evals[0] * (wb + wc) + self.mult_evals[0] * (wb * wc)
        } else {
            F::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::eval_from_evals;
    use crate::runner::sumcheck;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    /// Run GKR prover and verify the proof for a given k.
    fn run_gkr_test(k: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 1 << k;
        let n_bc = n * n;

        let add_evals: Vec<F64> = (0..n_bc).map(|_| F64::rand(&mut rng)).collect();
        let mult_evals: Vec<F64> = (0..n_bc).map(|_| F64::rand(&mut rng)).collect();
        let w_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Compute claimed sum directly.
        let mut expected_sum = F64::ZERO;
        for b in 0..n {
            for c in 0..n {
                let idx = b * n + c;
                let wb = w_evals[b];
                let wc = w_evals[c];
                expected_sum += add_evals[idx] * (wb + wc) + mult_evals[idx] * (wb * wc);
            }
        }

        // Run the prover.
        let mut prover = GkrProver::new(add_evals.clone(), mult_evals.clone(), w_evals.clone());
        let num_rounds = 2 * k;
        let mut trng = StdRng::seed_from_u64(99);
        let mut transcript = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, num_rounds, &mut transcript, |_, _| {});

        // Verify: q(0) + q(1) = claim each round, update via Lagrange.
        let mut claim = expected_sum;
        for (round, evals) in proof.round_polys.iter().enumerate() {
            assert_eq!(
                evals[0] + evals[1],
                claim,
                "k={k}: round {round}: q(0) + q(1) != claim"
            );
            claim = eval_from_evals(evals, proof.challenges[round]);
        }
        assert_eq!(claim, proof.final_value, "k={k}: final claim mismatch");

        // Verify claimed W values match multilinear extension.
        let (w_b_star, w_c_star) = prover.claimed_w_values();
        assert_eq!(
            w_b_star,
            eval_mle(&w_evals, &proof.challenges[..k]),
            "k={k}: W(b*) mismatch"
        );
        assert_eq!(
            w_c_star,
            eval_mle(&w_evals, &proof.challenges[k..]),
            "k={k}: W(c*) mismatch"
        );
    }

    /// Evaluate the multilinear extension of `evals` at point `r`.
    fn eval_mle(evals: &[F64], r: &[F64]) -> F64 {
        let mut table = evals.to_vec();
        for &ri in r {
            let half = table.len() / 2;
            for j in 0..half {
                table[j] = table[j] + ri * (table[j + half] - table[j]);
            }
            table.truncate(half);
        }
        table[0]
    }

    #[test]
    fn gkr_k1() {
        run_gkr_test(1, 0x100);
    }

    #[test]
    fn gkr_k2() {
        run_gkr_test(2, 0x200);
    }

    #[test]
    fn gkr_k3() {
        run_gkr_test(3, 0x300);
    }

    #[test]
    fn gkr_k4() {
        run_gkr_test(4, 0x400);
    }

    /// All-zero gates: sum should be zero regardless of witness.
    #[test]
    fn gkr_zero_gates() {
        let k = 2;
        let n = 1 << k;
        let n_bc = n * n;
        let mut rng = StdRng::seed_from_u64(0x500);

        let add_evals = vec![F64::ZERO; n_bc];
        let mult_evals = vec![F64::ZERO; n_bc];
        let w_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut prover = GkrProver::new(add_evals, mult_evals, w_evals);
        let mut trng = StdRng::seed_from_u64(99);
        let mut transcript = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, 2 * k, &mut transcript, |_, _| {});

        // All round polynomials should be zero.
        for evals in &proof.round_polys {
            for &v in evals {
                assert_eq!(v, F64::ZERO);
            }
        }
    }

    /// Add-only circuit (no mult gates): degree is still 2 but
    /// mult contributions are zero.
    #[test]
    fn gkr_add_only() {
        let k = 2;
        let n = 1 << k;
        let n_bc = n * n;
        let mut rng = StdRng::seed_from_u64(0x600);

        let add_evals: Vec<F64> = (0..n_bc).map(|_| F64::rand(&mut rng)).collect();
        let mult_evals = vec![F64::ZERO; n_bc];
        let w_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut expected_sum = F64::ZERO;
        for b in 0..n {
            for c in 0..n {
                expected_sum += add_evals[b * n + c] * (w_evals[b] + w_evals[c]);
            }
        }

        let mut prover = GkrProver::new(add_evals, mult_evals, w_evals);
        let mut trng = StdRng::seed_from_u64(99);
        let mut transcript = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, 2 * k, &mut transcript, |_, _| {});

        assert_eq!(
            proof.round_polys[0][0] + proof.round_polys[0][1],
            expected_sum,
        );
    }
}
