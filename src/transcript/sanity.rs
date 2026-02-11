use ark_ff::Field;
use ark_std::rand::Rng;

use crate::transcript::Transcript;

#[derive(Debug)]
pub struct SanityTranscript<'a, R> {
    pub rng: &'a mut R,
}

impl<'a, R> SanityTranscript<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self { rng }
    }
}

impl<'a, F, R> Transcript<F> for SanityTranscript<'a, R>
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
