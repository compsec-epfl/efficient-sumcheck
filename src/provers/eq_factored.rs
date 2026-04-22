//! Eq-factored sumcheck prover: `g(x) = eq(w, x) · p(x)`, degree 2.
//!
//! Implements [`SumcheckProver`] for the common eq-factored sumcheck
//!
//! ```text
//! ∑_{x ∈ {0,1}^v} eq(w, x) · p(x) = H
//! ```
//!
//! where `w` is a point in `F^v` fixed at setup and `p` is a multilinear
//! polynomial given by its evaluations over the Boolean hypercube.
//!
//! Eq-factored sumchecks appear in lookup arguments and in any reduction
//! that couples a public point `w` to a witness polynomial via the
//! multilinear equality predicate.
//!
//! # Space: Split-Value Optimization (BDDT25 Algorithm 5)
//!
//! The multilinear equality polynomial factors as a tensor product over
//! any split of the variables:
//!
//! ```text
//! eq(w, x) = eq(w_L, x_L) · eq(w_R, x_R),    w = (w_L, w_R), x = (x_L, x_R)
//! ```
//!
//! This prover splits `w` down the middle (`v_L = ⌊v/2⌋`, `v_R = v - v_L`)
//! and stores only the two half-tables `eq(w_L, ·)` and `eq(w_R, ·)` of
//! `2^{v_L}` and `2^{v_R}` entries respectively — `O(2^{v/2})` eq storage
//! instead of `O(2^v)`. Round-polynomial contributions are streamed
//! directly from the two half-tables without ever materializing their
//! product.
//!
//! During sumcheck (MSB layout), the first `v_L` rounds bind left-half
//! variables and fold `eq_L`; the remaining `v_R` rounds bind right-half
//! variables and fold `eq_R`. After the last round,
//! `eq(w, r) = eq_L[0] · eq_R[0]`.
//!
//! # Wire format
//!
//! EvalsInfty for degree 2: `[q(0), q(∞)]`. The verifier derives
//! `q(1) = claim - q(0)` and reconstructs the round polynomial.

use crate::field::SumcheckField;
use crate::inner_product_sumcheck as ip;
use crate::sumcheck_prover::SumcheckProver;

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// Eq-factored sumcheck prover for `∑_x eq(w, x) · p(x)` (degree 2).
///
/// Uses the Split-Value Optimization (BDDT25 Algorithm 5): `eq` is stored
/// as two half-tables `eq(w_L, ·)` and `eq(w_R, ·)` of `2^{v/2}` entries
/// each, rather than a single `2^v` table. Round polynomials are
/// assembled on the fly by a nested-sum kernel that reads the two
/// half-tables — see the module docs for the math.
///
/// # Construction
///
/// ```ignore
/// use effsc::provers::eq_factored::EqFactoredProver;
/// use effsc::runner::sumcheck;
///
/// let mut prover = EqFactoredProver::new(w, p_evals);
/// let proof = sumcheck(&mut prover, num_vars, &mut transcript, noop_hook);
/// // final_value() = eq(w, r) · p(r)
/// let (p_r, eq_wr) = prover.final_factors();
/// ```
pub struct EqFactoredProver<F: SumcheckField> {
    /// `p` evaluations (MSB layout), padded to `2^v` on construction.
    /// Folded in every round.
    p: Vec<F>,
    /// `eq(w_L, ·)` over `{0,1}^{v_L}` (MSB layout). Folded during
    /// left-half rounds; becomes a scalar after round `v_L - 1`.
    eq_l: Vec<F>,
    /// `eq(w_R, ·)` over `{0,1}^{v_R}` (MSB layout). Unchanged during
    /// left-half rounds; folded during right-half rounds.
    eq_r: Vec<F>,
    /// Number of left-half variables `v_L`. `v_R = v - v_L = v - (v/2)`.
    v_l: usize,
    /// Total number of variables.
    v: usize,
    /// Number of completed [`round`](SumcheckProver::round) calls.
    rounds_elapsed: usize,
}

impl<F: SumcheckField> EqFactoredProver<F> {
    /// Construct a prover for `∑_x eq(w, x) · p(x)`.
    ///
    /// `p_evals.len()` must be `≤ 2^{w.len()}`; shorter inputs are
    /// zero-padded to `2^{w.len()}`.
    pub fn new(w: Vec<F>, p_evals: Vec<F>) -> Self {
        let v = w.len();
        let n = 1usize << v;
        assert!(
            p_evals.len() <= n,
            "p_evals length {} exceeds 2^{} = {}",
            p_evals.len(),
            v,
            n
        );
        let v_l = v / 2;
        let (w_l, w_r) = w.split_at(v_l);
        let eq_l = build_eq_table(w_l);
        let eq_r = build_eq_table(w_r);
        let mut p = p_evals;
        p.resize(n, F::ZERO);
        Self {
            p,
            eq_l,
            eq_r,
            v_l,
            v,
            rounds_elapsed: 0,
        }
    }

    /// After full sumcheck: `(p(r), eq(w, r))`.
    pub fn final_factors(&self) -> (F, F) {
        if self.p.len() == 1 {
            (self.p[0], self.eq_l[0] * self.eq_r[0])
        } else {
            (F::ZERO, F::ZERO)
        }
    }

    /// Round polynomial during the left-half phase (`rounds_elapsed < v_L`).
    ///
    /// Let `j = rounds_elapsed` and split the remaining variables as
    /// `(x_j, a, b)` where `a ∈ {0,1}^{v_L − j − 1}` and `b ∈ {0,1}^{v_R}`.
    /// Then
    ///
    /// ```text
    /// q(x_j) = Σ_a eq_L(x_j, a) · Σ_b eq_R(b) · p(x_j, a, b).
    /// ```
    ///
    /// Splitting `eq_L = [eq_L_lo, eq_L_hi]` and `p = [p_lo, p_hi]` by the
    /// leading bit `x_j`, each `a` indexes a `2^{v_R}`-sized slice of both
    /// halves of `p`. The kernel contracts the `b` dimension against
    /// `eq_R` once per `a`, then accumulates the outer sum.
    fn round_poly_left(&self) -> Vec<F> {
        let eq_l_half = self.eq_l.len() >> 1;
        let (eq_l_lo, eq_l_hi) = self.eq_l.split_at(eq_l_half);

        let p_half = self.p.len() >> 1;
        let (p_lo, p_hi) = self.p.split_at(p_half);

        let m = self.eq_r.len();
        debug_assert_eq!(eq_l_half * m, p_half);

        let mut q0 = F::ZERO;
        let mut q_inf = F::ZERO;
        for a in 0..eq_l_half {
            let p_lo_slice = &p_lo[a * m..(a + 1) * m];
            let p_hi_slice = &p_hi[a * m..(a + 1) * m];

            let mut inner_0 = F::ZERO;
            let mut inner_delta = F::ZERO;
            for b in 0..m {
                let er = self.eq_r[b];
                let pl = p_lo_slice[b];
                let ph = p_hi_slice[b];
                inner_0 += er * pl;
                inner_delta += er * (ph - pl);
            }

            let el_lo = eq_l_lo[a];
            let el_hi = eq_l_hi[a];
            q0 += el_lo * inner_0;
            q_inf += (el_hi - el_lo) * inner_delta;
        }

        vec![q0, q_inf]
    }

    /// Round polynomial during the right-half phase
    /// (`rounds_elapsed ≥ v_L`). `eq_L` has already folded to a scalar;
    /// the remaining work is a standard inner-product round polynomial on
    /// `eq_R` and `p`, scaled by that scalar.
    fn round_poly_right(&self) -> Vec<F> {
        let n = self.p.len();
        debug_assert_eq!(self.eq_l.len(), 1);
        debug_assert_eq!(self.eq_r.len(), n);

        let scalar = self.eq_l[0];
        if n <= 1 {
            let v = if n == 1 { scalar * self.eq_r[0] * self.p[0] } else { F::ZERO };
            return vec![v, F::ZERO];
        }

        let half = n >> 1;
        let (eq_r_lo, eq_r_hi) = self.eq_r.split_at(half);
        let (p_lo, p_hi) = self.p.split_at(half);

        let mut q0 = F::ZERO;
        let mut q_inf = F::ZERO;
        for i in 0..half {
            let el = eq_r_lo[i];
            let eh = eq_r_hi[i];
            let pl = p_lo[i];
            let ph = p_hi[i];
            q0 += el * pl;
            q_inf += (eh - el) * (ph - pl);
        }

        vec![q0 * scalar, q_inf * scalar]
    }
}

/// Build the multilinear-extension table of `eq(w, ·)` over `{0,1}^v` in
/// MSB layout: `table[idx]` is indexed bit-by-bit MSB-first.
///
/// Runs in `O(2^v)` time with `O(2^v)` space. With `w.len() = v/2` this
/// produces one of the split-value half-tables.
pub(crate) fn build_eq_table<F: SumcheckField>(w: &[F]) -> Vec<F> {
    let v = w.len();
    if v == 0 {
        return vec![F::ONE];
    }
    let size = 1usize << v;
    let mut table = vec![F::ZERO; size];
    table[0] = F::ONE;
    // Process variables from most-significant to least-significant so that
    // the final layout has w[0] as the MSB (topmost bit of the index).
    for (j, &wj) in w.iter().enumerate() {
        let stride = 1usize << (v - 1 - j);
        let block = 2 * stride;
        let populated_blocks = 1usize << j;
        for b in 0..populated_blocks {
            let base = b * block;
            let parent = table[base];
            table[base] = parent * (F::ONE - wj);
            table[base + stride] = parent * wj;
        }
    }
    table
}

#[cfg(feature = "arkworks")]
impl<F> SumcheckProver<F> for EqFactoredProver<F>
where
    F: ark_ff::Field,
{
    fn degree(&self) -> usize {
        2
    }

    fn round(&mut self, challenge: Option<F>) -> Vec<F> {
        // Apply the previous round's challenge to the correct half of eq
        // (plus p, which folds every round).
        if let Some(r) = challenge {
            let prev_var = self.rounds_elapsed - 1;
            ip::fold(&mut self.p, r);
            if prev_var < self.v_l {
                ip::fold(&mut self.eq_l, r);
            } else {
                ip::fold(&mut self.eq_r, r);
            }
        }

        let j = self.rounds_elapsed;
        let round_poly = if j < self.v_l {
            self.round_poly_left()
        } else {
            self.round_poly_right()
        };
        self.rounds_elapsed += 1;
        round_poly
    }

    fn finalize(&mut self, last_challenge: F) {
        let last_var = self.v - 1;
        ip::fold(&mut self.p, last_challenge);
        if last_var < self.v_l {
            ip::fold(&mut self.eq_l, last_challenge);
        } else {
            ip::fold(&mut self.eq_r, last_challenge);
        }
    }

    fn final_value(&self) -> F {
        if self.p.len() == 1 {
            self.p[0] * self.eq_l[0] * self.eq_r[0]
        } else {
            F::ZERO
        }
    }
}

#[cfg(all(test, feature = "arkworks"))]
mod tests {
    use super::*;
    use crate::runner::sumcheck;
    use crate::tests::F64;
    use crate::transcript::SanityTranscript;
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    /// Evaluate multilinear `eq(w, x)` at Boolean `x` (as bit pattern).
    fn eq_at_boolean(w: &[F64], x_bits: usize) -> F64 {
        let mut acc = F64::from(1u64);
        let v = w.len();
        for j in 0..v {
            // MSB-first indexing: bit (v-1-j) of x_bits corresponds to w[j].
            let xj = (x_bits >> (v - 1 - j)) & 1;
            acc *= if xj == 1 { w[j] } else { F64::from(1u64) - w[j] };
        }
        acc
    }

    #[test]
    fn build_eq_table_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(0xE01);
        let v = 4;
        let w: Vec<F64> = (0..v).map(|_| F64::rand(&mut rng)).collect();
        let table = build_eq_table(&w);
        for idx in 0..(1 << v) {
            assert_eq!(
                table[idx],
                eq_at_boolean(&w, idx),
                "eq(w, x={idx:04b}) mismatch"
            );
        }
    }

    #[test]
    fn eq_table_sums_to_one() {
        let mut rng = StdRng::seed_from_u64(0xE02);
        let v = 5;
        let w: Vec<F64> = (0..v).map(|_| F64::rand(&mut rng)).collect();
        let table = build_eq_table(&w);
        let s: F64 = table.iter().copied().sum();
        assert_eq!(s, F64::from(1u64), "Σ_x eq(w, x) = 1");
    }

    #[test]
    fn eq_factored_prover_completes_and_verifies() {
        let mut rng = StdRng::seed_from_u64(0xE03);
        let v = 5;
        let n = 1usize << v;
        let w: Vec<F64> = (0..v).map(|_| F64::rand(&mut rng)).collect();
        let p_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Claimed sum: H = Σ_x eq(w, x) · p(x).
        let mut claimed_sum = F64::from(0u64);
        for x in 0..n {
            claimed_sum += eq_at_boolean(&w, x) * p_evals[x];
        }

        let mut prover = EqFactoredProver::new(w.clone(), p_evals.clone());
        let mut trng = StdRng::seed_from_u64(0xBEEF);
        let mut t = SanityTranscript::new(&mut trng);
        let proof = sumcheck(&mut prover, v, &mut t, |_, _| {});

        // EvalsInfty: degree-2 wire is [q(0), q(∞)].
        for rp in &proof.round_polys {
            assert_eq!(rp.len(), 2, "EvalsInfty degree-2 wire length");
        }

        // Replay reductions: q(r) = q(0) + r·(q(1) − q(0) − q(∞)) + q(∞)·r².
        let mut claim = claimed_sum;
        for (rp, &r) in proof.round_polys.iter().zip(&proof.challenges) {
            let q0 = rp[0];
            let q_inf = rp[1];
            let q1 = claim - q0;
            claim = q0 + r * (q1 - q0 - q_inf) + q_inf * r * r;
        }
        assert_eq!(claim, proof.final_value);

        // Final-factor sanity: final_value = p(r) · eq(w, r).
        let (p_r, eq_wr) = prover.final_factors();
        assert_eq!(proof.final_value, p_r * eq_wr);
    }

    #[test]
    fn eq_factored_matches_inner_product_with_eq_table() {
        use crate::provers::inner_product::InnerProductProver;

        let mut rng = StdRng::seed_from_u64(0xE04);
        let v = 4;
        let n = 1usize << v;
        let w: Vec<F64> = (0..v).map(|_| F64::rand(&mut rng)).collect();
        let p_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Run EqFactoredProver.
        let mut eq_prover = EqFactoredProver::new(w.clone(), p_evals.clone());
        let mut trng1 = StdRng::seed_from_u64(0xCAFE);
        let mut t1 = SanityTranscript::new(&mut trng1);
        let eq_proof = sumcheck(&mut eq_prover, v, &mut t1, |_, _| {});

        // Run InnerProductProver with p and explicitly-computed eq(w, ·).
        let eq_table = build_eq_table(&w);
        let mut ip_prover = InnerProductProver::new(p_evals, eq_table);
        let mut trng2 = StdRng::seed_from_u64(0xCAFE);
        let mut t2 = SanityTranscript::new(&mut trng2);
        let ip_proof = sumcheck(&mut ip_prover, v, &mut t2, |_, _| {});

        // Same transcript seed → same challenges, same wire, same final value.
        assert_eq!(eq_proof.challenges, ip_proof.challenges);
        assert_eq!(eq_proof.round_polys, ip_proof.round_polys);
        assert_eq!(eq_proof.final_value, ip_proof.final_value);
    }

    /// Split-value correctness across a range of `v`, including odd `v`
    /// (where `v_L ≠ v_R`) and small edge cases.
    #[test]
    fn eq_factored_matches_inner_product_various_v() {
        use crate::provers::inner_product::InnerProductProver;

        for &v in &[1usize, 2, 3, 4, 5, 6, 7, 8] {
            let mut rng = StdRng::seed_from_u64(0xE05 ^ v as u64);
            let n = 1usize << v;
            let w: Vec<F64> = (0..v).map(|_| F64::rand(&mut rng)).collect();
            let p_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

            let mut eq_prover = EqFactoredProver::new(w.clone(), p_evals.clone());
            let mut trng1 = StdRng::seed_from_u64(0xC0FFEE ^ v as u64);
            let mut t1 = SanityTranscript::new(&mut trng1);
            let eq_proof = sumcheck(&mut eq_prover, v, &mut t1, |_, _| {});

            let eq_table = build_eq_table(&w);
            let mut ip_prover = InnerProductProver::new(p_evals, eq_table);
            let mut trng2 = StdRng::seed_from_u64(0xC0FFEE ^ v as u64);
            let mut t2 = SanityTranscript::new(&mut trng2);
            let ip_proof = sumcheck(&mut ip_prover, v, &mut t2, |_, _| {});

            assert_eq!(eq_proof.challenges, ip_proof.challenges, "v={v}");
            assert_eq!(eq_proof.round_polys, ip_proof.round_polys, "v={v}");
            assert_eq!(eq_proof.final_value, ip_proof.final_value, "v={v}");

            // Eq storage is sub-linear: eq_l + eq_r ≤ 2·2^⌈v/2⌉.
            let (_p_r, eq_wr) = eq_prover.final_factors();
            // Sanity: eq(w, r) as a scalar equals the product of the two
            // half-table scalars.
            let mut expected_eq_wr = F64::from(1u64);
            for (j, &rj) in eq_proof.challenges.iter().enumerate() {
                expected_eq_wr *= rj * w[j] + (F64::from(1u64) - rj) * (F64::from(1u64) - w[j]);
            }
            assert_eq!(eq_wr, expected_eq_wr, "v={v}: eq(w,r) mismatch");
        }
    }
}
