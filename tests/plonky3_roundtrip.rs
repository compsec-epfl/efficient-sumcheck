//! Roundtrip test demonstrating effsc with a Plonky3 field.
//!
//! Shows that any ecosystem's field type works with the sumcheck library
//! via a thin `SumcheckField` impl — no arkworks dependency required.

use effsc::field::SumcheckField;
use effsc::noop_hook;
use effsc::provers::multilinear_lsb::MultilinearProverLSB;
use effsc::runner::sumcheck;
use effsc::transcript::{ProverTranscript, VerifierTranscript};
use effsc::verifier::sumcheck_verify;

use p3_field::integers::QuotientMap;
use p3_field::Field;
use p3_goldilocks::Goldilocks;

use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ─── Newtype wrapper ───────────────────────────────────────────────────────

/// Thin wrapper around Plonky3's Goldilocks to implement `SumcheckField`.
#[derive(Copy, Clone, Debug, PartialEq)]
struct P3Goldilocks(Goldilocks);

impl P3Goldilocks {
    fn new(val: u64) -> Self {
        Self(Goldilocks::from_int(val))
    }
}

impl fmt::Display for P3Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl Add for P3Goldilocks {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for P3Goldilocks {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for P3Goldilocks {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Neg for P3Goldilocks {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl AddAssign for P3Goldilocks {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for P3Goldilocks {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for P3Goldilocks {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Sum for P3Goldilocks {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl SumcheckField for P3Goldilocks {
    const ZERO: Self = Self(Goldilocks::new(0));
    const ONE: Self = Self(Goldilocks::new(1));

    fn from_u64(val: u64) -> Self {
        Self(Goldilocks::from_int(val))
    }

    fn inverse(&self) -> Option<Self> {
        self.0.try_inverse().map(Self)
    }
}

// ─── Minimal transcript ────────────────────────────────────────────────────

/// Deterministic transcript for the roundtrip test.
struct CounterTranscript {
    counter: u64,
    tape: Vec<P3Goldilocks>,
    cursor: usize,
}

impl CounterTranscript {
    fn new() -> Self {
        Self {
            counter: 1,
            tape: Vec::new(),
            cursor: 0,
        }
    }

    fn rewind(&mut self) {
        self.cursor = 0;
    }
}

impl ProverTranscript<P3Goldilocks> for CounterTranscript {
    fn send(&mut self, value: P3Goldilocks) {
        self.tape.push(value);
    }

    fn challenge(&mut self) -> P3Goldilocks {
        self.counter += 1;
        let c = P3Goldilocks::new(self.counter);
        self.tape.push(c);
        c
    }
}

impl VerifierTranscript<P3Goldilocks> for CounterTranscript {
    type Error = core::convert::Infallible;

    fn receive(&mut self) -> Result<P3Goldilocks, Self::Error> {
        let v = self.tape[self.cursor];
        self.cursor += 1;
        Ok(v)
    }

    fn challenge(&mut self) -> P3Goldilocks {
        let v = self.tape[self.cursor];
        self.cursor += 1;
        v
    }
}

// ─── Test ──────────────────────────────────────────────────────────────────

#[test]
fn plonky3_multilinear_roundtrip() {
    let num_vars = 4;
    let n = 1 << num_vars;

    // Create evaluations: f(x) = x + 1 for x in 0..16.
    let evals: Vec<P3Goldilocks> = (0..n).map(|i| P3Goldilocks::new(i as u64 + 1)).collect();
    let claimed_sum: P3Goldilocks = evals.iter().copied().sum();

    // Prove.
    let mut prover = MultilinearProverLSB::new(evals);
    let mut t = CounterTranscript::new();
    let proof = sumcheck(&mut prover, num_vars, &mut t, noop_hook);

    assert_eq!(proof.round_polys.len(), num_vars);
    assert_eq!(proof.challenges.len(), num_vars);

    // Verify.
    t.rewind();
    let result = sumcheck_verify(claimed_sum, 1, num_vars, &mut t, |_, _| Ok(()))
        .expect("verification should pass");

    assert_eq!(result.final_claim, proof.final_value);
}
