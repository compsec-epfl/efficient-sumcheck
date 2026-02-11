use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField};
use spongefish::{DefaultHash, ProverState};

use crate::transcript::Transcript;

/// Newtype wrapper around spongefish's [`ProverState`] so we can implement [`Transcript`].
///
/// Uses the codec-level API (`add_scalars` / `challenge_scalars`) which is compatible
/// with [`DomainSeparator`]'s `FieldDomainSeparator` builder methods.
pub struct SpongefishTranscript<R: RngCore + CryptoRng = ark_std::rand::rngs::StdRng>(
    pub ProverState<DefaultHash, u8, R>,
);

impl<F, R> Transcript<F> for SpongefishTranscript<R>
where
    F: Field,
    R: RngCore + CryptoRng,
{
    fn read(&mut self) -> F {
        let [v] = self.0.challenge_scalars::<1>().unwrap();
        v
    }

    fn write(&mut self, value: F) {
        self.0.add_scalars(&[value]).unwrap();
    }
}

// Optional helpers so it's easy to get the prover state back out.
impl<R> SpongefishTranscript<R>
where
    R: RngCore + CryptoRng,
{
    pub fn new(prover_state: ProverState<DefaultHash, u8, R>) -> Self {
        Self(prover_state)
    }
    pub fn into_inner(self) -> ProverState<DefaultHash, u8, R> {
        self.0
    }
    pub fn as_inner(&self) -> &ProverState<DefaultHash, u8, R> {
        &self.0
    }
    pub fn as_inner_mut(&mut self) -> &mut ProverState<DefaultHash, u8, R> {
        &mut self.0
    }
}
