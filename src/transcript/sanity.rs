use ark_ff::Field;
use ark_std::rand::Rng;

use crate::transcript::Transcript;

/// Test transcript: writes are no-ops, reads return random challenges from the RNG.
#[derive(Debug)]
pub struct TestTranscript<'a, R> {
    pub rng: &'a mut R,
}

impl<'a, R> TestTranscript<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self { rng }
    }
}

impl<'a, F, R> Transcript<F> for TestTranscript<'a, R>
where
    F: Field,
    R: Rng,
{
    fn write(&mut self, _value: F) {
        // no-op
    }

    fn read(&mut self) -> F {
        F::rand(&mut self.rng)
    }
}

// Keep the old name as a type alias for backwards compatibility.
pub type SanityTranscript<'a, R> = TestTranscript<'a, R>;
