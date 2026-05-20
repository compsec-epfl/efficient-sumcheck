use rand_core::RngCore;

use crate::field::SumcheckField;
use crate::transcript::{ProverTranscript, VerifierTranscript};

/// Test transcript: sends are no-ops, receives return `Ok(random)`,
/// challenges return random values from the RNG.
///
/// Implements both [`ProverTranscript`] and [`VerifierTranscript`] for
/// convenience in tests that run prover + verifier with the same RNG.
#[derive(Debug)]
pub struct TestTranscript<'a, R> {
    pub rng: &'a mut R,
}

impl<'a, R> TestTranscript<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self { rng }
    }
}

// Randomness is `from_u64(rng.next_u64())` — base-field-width lifted into
// extensions. Adequate for a test transcript whose only soundness need is
// avoiding accidental collisions.
impl<'a, F, R> ProverTranscript<F> for TestTranscript<'a, R>
where
    F: SumcheckField,
    R: RngCore,
{
    fn send(&mut self, _value: F) {
        // no-op
    }

    fn challenge(&mut self) -> F {
        F::from_u64(self.rng.next_u64())
    }
}

impl<'a, F, R> VerifierTranscript<F> for TestTranscript<'a, R>
where
    F: SumcheckField,
    R: RngCore,
{
    type Error = core::convert::Infallible;

    fn receive(&mut self) -> Result<F, Self::Error> {
        Ok(F::from_u64(self.rng.next_u64()))
    }

    fn challenge(&mut self) -> F {
        F::from_u64(self.rng.next_u64())
    }
}

pub type SanityTranscript<'a, R> = TestTranscript<'a, R>;
