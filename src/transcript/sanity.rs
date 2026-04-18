use ark_ff::Field;
use ark_std::rand::Rng;

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

impl<'a, F, R> ProverTranscript<F> for TestTranscript<'a, R>
where
    F: Field,
    R: Rng,
{
    fn send(&mut self, _value: F) {
        // no-op
    }

    fn challenge(&mut self) -> F {
        F::rand(&mut self.rng)
    }
}

impl<'a, F, R> VerifierTranscript<F> for TestTranscript<'a, R>
where
    F: Field,
    R: Rng,
{
    type Error = core::convert::Infallible;

    fn receive(&mut self) -> Result<F, Self::Error> {
        Ok(F::rand(&mut self.rng))
    }

    fn challenge(&mut self) -> F {
        F::rand(&mut self.rng)
    }
}

pub type SanityTranscript<'a, R> = TestTranscript<'a, R>;
