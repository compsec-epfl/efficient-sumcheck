use ark_std::rand::{self, CryptoRng, RngCore};
use spongefish::{Decoding, DuplexSpongeInterface, Encoding, NargSerialize, ProverState};

use crate::experimental::transcript::Transcript;

/// Local “newtype” wrapper so we can implement a foreign trait.
pub struct SpongefishTranscript<
    H: DuplexSpongeInterface = spongefish::StdHash,
    R: RngCore + CryptoRng = rand::rngs::StdRng,
>(
    pub ProverState<H, R>,
);

impl<H, R, F> Transcript<F> for SpongefishTranscript<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    F: Encoding<[H::U]> + NargSerialize + Decoding<[H::U]>,
{
    fn read(&mut self) -> F {
        self.0.verifier_message::<F>()
    }

    fn write(&mut self, value: F) {
        self.0.prover_message(&value);
    }
}

// Optional helpers so it’s easy to get the prover state back out.
impl<H, R> SpongefishTranscript<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    pub fn into_inner(self) -> ProverState<H, R> {
        self.0
    }
    pub fn as_inner(&self) -> &ProverState<H, R> {
        &self.0
    }
    pub fn as_inner_mut(&mut self) -> &mut ProverState<H, R> {
        &mut self.0
    }
}
