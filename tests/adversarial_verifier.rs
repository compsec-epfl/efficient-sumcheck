//! Adversarial tests for `sumcheck_verify`.
//!
//! Verifies that honest proofs are accepted and corrupted proofs are rejected
//! across all three prover types: multilinear, inner product, and GKR.

use ark_ff::{AdditiveGroup, UniformRand};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use effsc::noop_hook_verify;
use effsc::proof::{SumcheckError, SumcheckProof};
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

#[test]
fn multilinear_corrupted_round_poly_rejected() {
    let mut t = ReplayTranscript::new(42);
    let (claimed_sum, _proof) = make_multilinear_proof(6, &mut t);

    // Corrupt the first evaluation of round 2 in the tape.
    // Each round for degree-1: 2 evals + 1 challenge = 3 elements.
    // Round 2 starts at offset 6, corrupt index 6.
    t.tape[6] += F64::from(1u64);

    t.rewind();
    let result = sumcheck_verify(claimed_sum, 1, 6, &mut t, noop_hook_verify);
    assert!(result.is_err());
    match result.unwrap_err() {
        SumcheckError::ConsistencyCheck { round } => assert_eq!(round, 2),
        e => panic!("expected ConsistencyCheck at round 2, got {e:?}"),
    }
}

// ─── Multilinear: wrong claimed sum ───────────────────────────────────────

#[test]
fn multilinear_wrong_claimed_sum_rejected() {
    let mut t = ReplayTranscript::new(42);
    let (claimed_sum, _proof) = make_multilinear_proof(6, &mut t);

    t.rewind();
    let result = sumcheck_verify(
        claimed_sum + F64::from(1u64), // wrong sum
        1,
        6,
        &mut t,
        noop_hook_verify,
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SumcheckError::ConsistencyCheck { round } => assert_eq!(round, 0),
        e => panic!("expected ConsistencyCheck at round 0, got {e:?}"),
    }
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
    let (claimed_sum, _proof) = make_inner_product_proof(6, &mut t);

    // Degree-2: 3 evals + 1 challenge = 4 elements per round.
    // Corrupt round 1, first eval at offset 4.
    t.tape[4] += F64::from(1u64);

    t.rewind();
    let result = sumcheck_verify(claimed_sum, 2, 6, &mut t, noop_hook_verify);
    assert!(result.is_err());
    match result.unwrap_err() {
        SumcheckError::ConsistencyCheck { round } => assert_eq!(round, 1),
        e => panic!("expected ConsistencyCheck at round 1, got {e:?}"),
    }
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
        let (claimed_sum, _proof, _prover) = make_gkr_proof(k, &mut t);

        // Degree-2: 3 evals + 1 challenge = 4 elements per round.
        // Corrupt round 3, first eval at offset 12.
        t.tape[12] += F64::from(1u64);

        t.rewind();
        let result = sumcheck_verify(claimed_sum, 2, 2 * k, &mut t, noop_hook_verify);
        assert!(result.is_err());
        match result.unwrap_err() {
            SumcheckError::ConsistencyCheck { round } => assert_eq!(round, 3),
            e => panic!("expected ConsistencyCheck at round 3, got {e:?}"),
        }
    }

    #[test]
    fn gkr_wrong_claimed_sum_rejected() {
        let k = 3;
        let mut t = ReplayTranscript::new(99);
        let (claimed_sum, _proof, _prover) = make_gkr_proof(k, &mut t);

        t.rewind();
        let result = sumcheck_verify(
            claimed_sum + F64::from(1u64),
            2,
            2 * k,
            &mut t,
            noop_hook_verify,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            SumcheckError::ConsistencyCheck { round } => assert_eq!(round, 0),
            e => panic!("expected ConsistencyCheck at round 0, got {e:?}"),
        }
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
