//! Adversarial tests for `sumcheck_verify`.
//!
//! Verifies that honest proofs are accepted and corrupted proofs are rejected
//! across all three prover types: multilinear, inner product, and GKR.

use ark_ff::{AdditiveGroup, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use effsc::noop_hook_verify;
use effsc::proof::SumcheckProof;
use effsc::provers::inner_product::InnerProductProver;
use effsc::provers::multilinear::MultilinearProver;
use effsc::runner::sumcheck;
use effsc::tests::F64;
use effsc::transcript::{ProverTranscript, VerifierTranscript};
use effsc::verifier::sumcheck_verify;

// ─── Replay transcript ────────────────────────────────────────────────────
//
// Records prover messages and challenges during proving, then replays them
// to sumcheck_verify. Both sides see the same sequence of field elements.

struct ReplayTranscript {
    /// Recorded field elements (round poly evals interleaved with challenges).
    tape: Vec<F64>,
    /// Current read position for the verifier.
    cursor: usize,
    /// RNG for generating challenges.
    rng: StdRng,
}

impl ReplayTranscript {
    fn new(seed: u64) -> Self {
        Self {
            tape: Vec::new(),
            cursor: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reset the cursor to replay from the beginning.
    fn rewind(&mut self) {
        self.cursor = 0;
    }
}

impl ProverTranscript<F64> for ReplayTranscript {
    fn send(&mut self, value: F64) {
        self.tape.push(value);
    }

    fn challenge(&mut self) -> F64 {
        let c = F64::rand(&mut self.rng);
        self.tape.push(c);
        c
    }
}

impl VerifierTranscript<F64> for ReplayTranscript {
    type Error = core::convert::Infallible;

    fn receive(&mut self) -> Result<F64, Self::Error> {
        let v = self.tape[self.cursor];
        self.cursor += 1;
        Ok(v)
    }

    fn challenge(&mut self) -> F64 {
        let v = self.tape[self.cursor];
        self.cursor += 1;
        v
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const SEED: u64 = 0xAD0E_0001;

fn make_multilinear_proof(
    num_vars: usize,
    transcript: &mut ReplayTranscript,
) -> (F64, SumcheckProof<F64>) {
    let n = 1 << num_vars;
    let mut rng = StdRng::seed_from_u64(SEED);
    let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
    let claimed_sum: F64 = evals.iter().copied().sum();

    let mut prover = MultilinearProver::new(evals);
    let proof = sumcheck(&mut prover, num_vars, transcript, |_, _| {});
    (claimed_sum, proof)
}

fn make_inner_product_proof(
    num_vars: usize,
    transcript: &mut ReplayTranscript,
) -> (F64, SumcheckProof<F64>) {
    let n = 1 << num_vars;
    let mut rng = StdRng::seed_from_u64(SEED);
    let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
    let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
    let claimed_sum: F64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();

    let mut prover = InnerProductProver::new(a, b);
    let proof = sumcheck(&mut prover, num_vars, transcript, |_, _| {});
    (claimed_sum, proof)
}

// ─── Multilinear: honest accept ───────────────────────────────────────────

#[test]
fn multilinear_honest_proof_accepted() {
    let mut t = ReplayTranscript::new(42);
    let (claimed_sum, proof) = make_multilinear_proof(6, &mut t);

    t.rewind();
    let result = sumcheck_verify(claimed_sum, 1, 6, &mut t, noop_hook_verify);
    assert!(result.is_ok());
    let r = result.unwrap();
    assert_eq!(r.final_claim, proof.final_value);
}

// ─── Multilinear: corrupted round poly ────────────────────────────────────
//
// Under the EvalsInfty wire format, the consistency check `h(0) + h(1) = claim`
// is structural (h(1) is derived from claim), so corruption is not caught by
// an early `ConsistencyCheck` error. Instead, the `final_claim` returned by
// the verifier diverges from the prover's `final_value`, and the caller's
// oracle check catches the discrepancy. Soundness is preserved; detection
// moves from the per-round consistency check to the final oracle check.

#[test]
fn multilinear_corrupted_round_poly_rejected() {
    let mut t = ReplayTranscript::new(42);
    let (claimed_sum, proof) = make_multilinear_proof(6, &mut t);

    // Degree-1 EvalsInfty: 1 eval + 1 challenge = 2 tape elements per round.
    // Round 3 begins at offset 6 — corrupt h(0) of round 3.
    t.tape[6] += F64::from(1u64);

    t.rewind();
    let result = sumcheck_verify(claimed_sum, 1, 6, &mut t, noop_hook_verify);
    // Verifier accepts (no per-round consistency error), but final_claim
    // diverges from the honest prover's final_value — the oracle check
    // catches the corruption.
    let r = result.expect("verifier returns Ok even under corruption");
    assert_ne!(r.final_claim, proof.final_value);
}

// ─── Multilinear: wrong claimed sum ───────────────────────────────────────

#[test]
fn multilinear_wrong_claimed_sum_rejected() {
    let mut t = ReplayTranscript::new(42);
    let (claimed_sum, proof) = make_multilinear_proof(6, &mut t);

    t.rewind();
    let result = sumcheck_verify(
        claimed_sum + F64::from(1u64), // wrong sum
        1,
        6,
        &mut t,
        noop_hook_verify,
    );
    // Under EvalsInfty, the wrong claim propagates into derived h(1) values
    // and the reconstructed round polynomials. Verifier does not reject
    // mid-protocol; the final_claim disagrees with proof.final_value, which
    // the caller's oracle check catches.
    let r = result.expect("verifier returns Ok even with wrong claimed sum");
    assert_ne!(r.final_claim, proof.final_value);
}

// ─── Multilinear: caller catches wrong final value ────────────────────────

#[test]
fn multilinear_caller_catches_wrong_final_value() {
    let mut t = ReplayTranscript::new(42);
    let (claimed_sum, proof) = make_multilinear_proof(6, &mut t);

    t.rewind();
    let r = sumcheck_verify(claimed_sum, 1, 6, &mut t, noop_hook_verify).unwrap();

    // The verifier returns the final claim. The caller checks it.
    assert_eq!(r.final_claim, proof.final_value);

    // If the prover lied about final_value, the caller catches it:
    let wrong_value = proof.final_value + F64::from(1u64);
    assert_ne!(r.final_claim, wrong_value);
}

// ─── Inner product: honest accept ─────────────────────────────────────────

#[test]
fn inner_product_honest_proof_accepted() {
    let mut t = ReplayTranscript::new(77);
    let (claimed_sum, proof) = make_inner_product_proof(6, &mut t);

    t.rewind();
    let r = sumcheck_verify(claimed_sum, 2, 6, &mut t, noop_hook_verify).unwrap();
    assert_eq!(r.final_claim, proof.final_value);
}

// ─── Inner product: corrupted round poly ──────────────────────────────────

#[test]
fn inner_product_corrupted_round_poly_rejected() {
    let mut t = ReplayTranscript::new(77);
    let (claimed_sum, proof) = make_inner_product_proof(6, &mut t);

    // Degree-2 EvalsInfty: 2 evals + 1 challenge = 3 tape elements per round.
    // Corrupt q(0) of round 1 at offset 3.
    t.tape[3] += F64::from(1u64);

    t.rewind();
    let result = sumcheck_verify(claimed_sum, 2, 6, &mut t, noop_hook_verify);
    // Verifier accepts; oracle check catches the corruption via final_claim.
    let r = result.expect("verifier returns Ok even under corruption");
    assert_ne!(r.final_claim, proof.final_value);
}

// ─── Inner product: caller catches wrong final value ──────────────────────

#[test]
fn inner_product_caller_catches_wrong_final_value() {
    let mut t = ReplayTranscript::new(77);
    let (claimed_sum, proof) = make_inner_product_proof(6, &mut t);

    t.rewind();
    let r = sumcheck_verify(claimed_sum, 2, 6, &mut t, noop_hook_verify).unwrap();
    assert_eq!(r.final_claim, proof.final_value);

    let wrong_value = proof.final_value + F64::from(1u64);
    assert_ne!(r.final_claim, wrong_value);
}

// ─── GKR: adversarial tests with deferred oracle check ───────────────────

#[cfg(feature = "arkworks")]
mod gkr_tests {
    use super::*;
    use effsc::provers::gkr::GkrProver;

    const GKR_SEED: u64 = 0xBEEF_0A70;

    fn make_gkr_proof(
        k: usize,
        transcript: &mut ReplayTranscript,
    ) -> (F64, SumcheckProof<F64>, GkrProver<F64>) {
        let n = 1 << k;
        let n_bc = n * n;
        let mut rng = StdRng::seed_from_u64(GKR_SEED);

        let add_evals: Vec<F64> = (0..n_bc).map(|_| F64::rand(&mut rng)).collect();
        let mult_evals: Vec<F64> = (0..n_bc).map(|_| F64::rand(&mut rng)).collect();
        let w_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut expected_sum = F64::ZERO;
        for b in 0..n {
            for c in 0..n {
                let idx = b * n + c;
                let wb = w_evals[b];
                let wc = w_evals[c];
                expected_sum += add_evals[idx] * (wb + wc) + mult_evals[idx] * (wb * wc);
            }
        }

        let mut prover = GkrProver::new(add_evals, mult_evals, w_evals);
        let num_rounds = 2 * k;
        let proof = sumcheck(&mut prover, num_rounds, transcript, |_, _| {});
        (expected_sum, proof, prover)
    }

    #[test]
    fn gkr_honest_proof_accepted() {
        let k = 3;
        let mut t = ReplayTranscript::new(99);
        let (claimed_sum, proof, _prover) = make_gkr_proof(k, &mut t);

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 2, 2 * k, &mut t, noop_hook_verify).unwrap();

        // GKR: final_claim is correct, caller passes it to the next layer.
        assert_eq!(r.final_claim, proof.final_value);
    }

    #[test]
    fn gkr_corrupted_round_poly_rejected() {
        let k = 3;
        let mut t = ReplayTranscript::new(99);
        let (claimed_sum, proof, _prover) = make_gkr_proof(k, &mut t);

        // Degree-2 EvalsInfty: 2 evals + 1 challenge = 3 tape elements per round.
        // Corrupt q(0) of round 3 at offset 9.
        t.tape[9] += F64::from(1u64);

        t.rewind();
        let result = sumcheck_verify(claimed_sum, 2, 2 * k, &mut t, noop_hook_verify);
        let r = result.expect("verifier returns Ok even under corruption");
        assert_ne!(r.final_claim, proof.final_value);
    }

    #[test]
    fn gkr_wrong_claimed_sum_rejected() {
        let k = 3;
        let mut t = ReplayTranscript::new(99);
        let (claimed_sum, proof, _prover) = make_gkr_proof(k, &mut t);

        t.rewind();
        let result = sumcheck_verify(
            claimed_sum + F64::from(1u64),
            2,
            2 * k,
            &mut t,
            noop_hook_verify,
        );
        let r = result.expect("verifier returns Ok even with wrong claimed sum");
        assert_ne!(r.final_claim, proof.final_value);
    }

    /// Demonstrates that the caller is responsible for the oracle check.
    /// sumcheck_verify returns final_claim; the caller must verify it.
    #[test]
    fn gkr_caller_must_check_final_claim() {
        let k = 3;
        let mut t = ReplayTranscript::new(99);
        let (claimed_sum, proof, _prover) = make_gkr_proof(k, &mut t);

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 2, 2 * k, &mut t, noop_hook_verify).unwrap();

        // The verifier gives the caller the final_claim.
        // A composed caller (GKR) passes it to the next layer.
        // A standalone caller checks it directly:
        assert_eq!(r.final_claim, proof.final_value);

        // If the caller forgets to check, a lying prover could claim
        // a different final_value. The round checks still pass — only
        // the oracle check catches it.
        let lying_final_value = proof.final_value + F64::from(1u64);
        assert_ne!(r.final_claim, lying_final_value);
    }
}

// ─── EqFactoredProver: end-to-end through sumcheck_verify ─────────────────
//
// Closes a gap: EqFactoredProver's unit tests cross-validate against
// InnerProductProver but never exercise the actual verifier.

#[cfg(feature = "arkworks")]
mod eq_factored_tests {
    use super::*;
    use effsc::provers::eq_factored::EqFactoredProver;

    const EQ_SEED: u64 = 0xEC_FA_C7_ED;

    fn make_eq_factored_proof(
        v: usize,
        transcript: &mut ReplayTranscript,
    ) -> (F64, SumcheckProof<F64>) {
        let n = 1usize << v;
        let mut rng = StdRng::seed_from_u64(EQ_SEED);
        let w: Vec<F64> = (0..v).map(|_| F64::rand(&mut rng)).collect();
        let p_evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Claimed sum: H = Σ_x eq(w, x) · p(x), computed brute-force.
        let eq_at_boolean = |x_bits: usize| -> F64 {
            let mut acc = F64::from(1u64);
            for j in 0..v {
                let xj = (x_bits >> (v - 1 - j)) & 1;
                acc *= if xj == 1 {
                    w[j]
                } else {
                    F64::from(1u64) - w[j]
                };
            }
            acc
        };
        let claimed_sum: F64 = (0..n).map(|x| eq_at_boolean(x) * p_evals[x]).sum();

        let mut prover = EqFactoredProver::new(w, p_evals);
        let proof = sumcheck(&mut prover, v, transcript, |_, _| {});
        (claimed_sum, proof)
    }

    #[test]
    fn eq_factored_honest_proof_accepted() {
        let v = 5;
        let mut t = ReplayTranscript::new(0xE0);
        let (claimed_sum, proof) = make_eq_factored_proof(v, &mut t);

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 2, v, &mut t, noop_hook_verify).unwrap();
        assert_eq!(r.final_claim, proof.final_value);
    }

    #[test]
    fn eq_factored_corrupted_round_poly_rejected() {
        let v = 5;
        let mut t = ReplayTranscript::new(0xE0);
        let (claimed_sum, proof) = make_eq_factored_proof(v, &mut t);

        // Degree-2 EvalsInfty: 2 evals + 1 challenge per round. Corrupt q(∞)
        // of round 2 at offset 2·3 + 1 = 7.
        t.tape[7] += F64::from(1u64);

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 2, v, &mut t, noop_hook_verify)
            .expect("verifier returns Ok; oracle check catches it");
        assert_ne!(r.final_claim, proof.final_value);
    }

    #[test]
    fn eq_factored_wrong_claimed_sum_rejected() {
        let v = 5;
        let mut t = ReplayTranscript::new(0xE0);
        let (claimed_sum, proof) = make_eq_factored_proof(v, &mut t);

        t.rewind();
        let r = sumcheck_verify(
            claimed_sum + F64::from(1u64),
            2,
            v,
            &mut t,
            noop_hook_verify,
        )
        .expect("verifier returns Ok; oracle check catches it");
        assert_ne!(r.final_claim, proof.final_value);
    }
}

// ─── CoefficientProver (d = 3): end-to-end through sumcheck_verify ────────
//
// Closes a gap: no prover ever exercised the verifier's polynomial
// reconstruction path for `d >= 3` before this.

#[cfg(feature = "arkworks")]
mod coefficient_degree3_tests {
    use super::*;
    use effsc::coefficient_sumcheck::RoundPolyEvaluator;
    use effsc::provers::coefficient::CoefficientProver;

    /// Round-polynomial evaluator for the degree-3 sumcheck
    /// `Σ_x a(x) · b(x) · c(x)`. Accumulates coefficients of the univariate
    /// round polynomial from the three pairwise tables.
    struct TripleProductEval;
    impl RoundPolyEvaluator<F64> for TripleProductEval {
        fn degree(&self) -> usize {
            3
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (la, ha) = pw[0];
            let (lb, hb) = pw[1];
            let (lc, hc) = pw[2];
            let da = ha - la;
            let db = hb - lb;
            let dc = hc - lc;
            // (la + x·da)(lb + x·db)(lc + x·dc)
            coeffs[0] += la * lb * lc;
            coeffs[1] += la * lb * dc + la * db * lc + da * lb * lc;
            coeffs[2] += la * db * dc + da * lb * dc + da * db * lc;
            coeffs[3] += da * db * dc;
        }
        fn parallelize(&self) -> bool {
            false
        }
    }

    const TRIPLE_SEED: u64 = 0xABC_D3F03;

    fn make_triple_product_proof(
        v: usize,
        transcript: &mut ReplayTranscript,
    ) -> (F64, SumcheckProof<F64>) {
        let n = 1usize << v;
        let mut rng = StdRng::seed_from_u64(TRIPLE_SEED);
        let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let c: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = (0..n).map(|i| a[i] * b[i] * c[i]).sum();

        let evaluator = TripleProductEval;
        let pairwise = vec![a, b, c];
        let tablewise: Vec<Vec<Vec<F64>>> = vec![];
        // CoefficientProver borrows `evaluator`, so we leak it here to match
        // the test-helper lifetime (same trick as keeping `prover` alive in
        // the test body).
        let evaluator_ref: &'static TripleProductEval = Box::leak(Box::new(evaluator));
        let mut prover = CoefficientProver::new(evaluator_ref, tablewise, pairwise);
        let proof = sumcheck(&mut prover, v, transcript, |_, _| {});
        (claimed_sum, proof)
    }

    #[test]
    fn degree3_honest_proof_accepted() {
        let v = 5;
        let mut t = ReplayTranscript::new(0xD3);
        let (claimed_sum, proof) = make_triple_product_proof(v, &mut t);

        // Every round's wire must carry exactly `d = 3` values.
        for (i, rp) in proof.round_polys.iter().enumerate() {
            assert_eq!(rp.len(), 3, "round {i}: EvalsInfty degree-3 wire length");
        }

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 3, v, &mut t, noop_hook_verify).unwrap();
        assert_eq!(r.final_claim, proof.final_value);
    }

    #[test]
    fn degree3_corrupted_round_poly_rejected() {
        let v = 5;
        let mut t = ReplayTranscript::new(0xD3);
        let (claimed_sum, proof) = make_triple_product_proof(v, &mut t);

        // Degree-3 EvalsInfty: 3 evals + 1 challenge = 4 tape elements per round.
        // Corrupt h(2) of round 1 at offset 4 + 2 = 6.
        t.tape[6] += F64::from(1u64);

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 3, v, &mut t, noop_hook_verify)
            .expect("verifier returns Ok; oracle check catches it");
        assert_ne!(r.final_claim, proof.final_value);
    }
}

// ─── LSB variants: end-to-end through sumcheck_verify ─────────────────────
//
// Wire format is independent of MSB vs LSB variable ordering — the bytes
// on the wire are identical shape. These tests confirm that the LSB
// provers' EvalsInfty emission matches the verifier's expectations.

#[cfg(feature = "arkworks")]
mod inner_product_lsb_tests {
    use super::*;
    use effsc::provers::inner_product_lsb::InnerProductProverLSB;

    const IP_LSB_SEED: u64 = 0x1B_50_BE_EF;

    fn make_ip_lsb_proof(
        num_vars: usize,
        transcript: &mut ReplayTranscript,
    ) -> (F64, SumcheckProof<F64>) {
        let n = 1 << num_vars;
        let mut rng = StdRng::seed_from_u64(IP_LSB_SEED);
        let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();

        let mut prover = InnerProductProverLSB::new(a, b);
        let proof = sumcheck(&mut prover, num_vars, transcript, |_, _| {});
        (claimed_sum, proof)
    }

    #[test]
    fn inner_product_lsb_honest_proof_accepted() {
        let v = 6;
        let mut t = ReplayTranscript::new(0x1B01);
        let (claimed_sum, proof) = make_ip_lsb_proof(v, &mut t);

        for (i, rp) in proof.round_polys.iter().enumerate() {
            assert_eq!(rp.len(), 2, "round {i}: EvalsInfty degree-2 wire length");
        }

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 2, v, &mut t, noop_hook_verify).unwrap();
        assert_eq!(r.final_claim, proof.final_value);
    }

    #[test]
    fn inner_product_lsb_corrupted_round_poly_rejected() {
        let v = 6;
        let mut t = ReplayTranscript::new(0x1B01);
        let (claimed_sum, proof) = make_ip_lsb_proof(v, &mut t);

        // Degree-2 EvalsInfty: 2 evals + 1 challenge = 3 tape elements per round.
        // Corrupt q(0) of round 2 at offset 6.
        t.tape[6] += F64::from(1u64);

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 2, v, &mut t, noop_hook_verify)
            .expect("verifier returns Ok; oracle check catches it");
        assert_ne!(r.final_claim, proof.final_value);
    }
}

#[cfg(feature = "arkworks")]
mod coefficient_lsb_degree3_tests {
    use super::*;
    use effsc::coefficient_sumcheck::RoundPolyEvaluator;
    use effsc::provers::coefficient_lsb::CoefficientProverLSB;

    /// Same triple-product evaluator as the MSB degree-3 test — the wire
    /// format is shape-identical across variable orderings, so only the
    /// prover constructor differs.
    struct TripleProductEval;
    impl RoundPolyEvaluator<F64> for TripleProductEval {
        fn degree(&self) -> usize {
            3
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (la, ha) = pw[0];
            let (lb, hb) = pw[1];
            let (lc, hc) = pw[2];
            let da = ha - la;
            let db = hb - lb;
            let dc = hc - lc;
            coeffs[0] += la * lb * lc;
            coeffs[1] += la * lb * dc + la * db * lc + da * lb * lc;
            coeffs[2] += la * db * dc + da * lb * dc + da * db * lc;
            coeffs[3] += da * db * dc;
        }
        fn parallelize(&self) -> bool {
            false
        }
    }

    const TRIPLE_LSB_SEED: u64 = 0xABC_D3F0_1B;

    fn make_triple_product_lsb_proof(
        v: usize,
        transcript: &mut ReplayTranscript,
    ) -> (F64, SumcheckProof<F64>) {
        let n = 1usize << v;
        let mut rng = StdRng::seed_from_u64(TRIPLE_LSB_SEED);
        let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let c: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = (0..n).map(|i| a[i] * b[i] * c[i]).sum();

        let evaluator: &'static TripleProductEval = Box::leak(Box::new(TripleProductEval));
        let pairwise = vec![a, b, c];
        let tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut prover = CoefficientProverLSB::new(evaluator, tablewise, pairwise);
        let proof = sumcheck(&mut prover, v, transcript, |_, _| {});
        (claimed_sum, proof)
    }

    #[test]
    fn degree3_lsb_honest_proof_accepted() {
        let v = 5;
        let mut t = ReplayTranscript::new(0xD3_1B);
        let (claimed_sum, proof) = make_triple_product_lsb_proof(v, &mut t);

        for (i, rp) in proof.round_polys.iter().enumerate() {
            assert_eq!(rp.len(), 3, "round {i}: EvalsInfty degree-3 wire length");
        }

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 3, v, &mut t, noop_hook_verify).unwrap();
        assert_eq!(r.final_claim, proof.final_value);
    }

    #[test]
    fn degree3_lsb_corrupted_round_poly_rejected() {
        let v = 5;
        let mut t = ReplayTranscript::new(0xD3_1B);
        let (claimed_sum, proof) = make_triple_product_lsb_proof(v, &mut t);

        // Degree-3 EvalsInfty: 3 evals + 1 challenge = 4 tape elements per round.
        // Corrupt h(∞) of round 1 at offset 4 + 1 = 5.
        t.tape[5] += F64::from(1u64);

        t.rewind();
        let r = sumcheck_verify(claimed_sum, 3, v, &mut t, noop_hook_verify)
            .expect("verifier returns Ok; oracle check catches it");
        assert_ne!(r.final_claim, proof.final_value);
    }
}
