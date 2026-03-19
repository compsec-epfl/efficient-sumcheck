use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::{Decoding, Encoding, ProverState, StdHash};

use crate::transcript::Transcript;

/// Newtype wrapper around spongefish's [`ProverState`] so we can implement [`Transcript`].
///
/// Uses the codec-level API (`prover_message` / `verifier_message`) which is compatible
/// with the new spongefish `domain_separator!` macro.
pub struct SpongefishTranscript<R: RngCore + CryptoRng = ark_std::rand::rngs::StdRng>(
    pub ProverState<StdHash, R>,
);

impl<F, R> Transcript<F> for SpongefishTranscript<R>
where
    F: Field + Encoding<[u8]> + Decoding<[u8]>,
    R: RngCore + CryptoRng,
{
    fn read(&mut self) -> F {
        self.0.verifier_message::<F>()
    }

    fn write(&mut self, value: F) {
        self.0.prover_message(&value);
    }
}

/// Blanket impl so raw `ProverState` can be used as a `Transcript` directly.
impl<F, H, R> Transcript<F> for spongefish::ProverState<H, R>
where
    F: Field + Encoding<[H::U]> + Decoding<[H::U]> + spongefish::NargSerialize,
    H: spongefish::DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    fn read(&mut self) -> F {
        self.verifier_message::<F>()
    }
    fn write(&mut self, value: F) {
        self.prover_message(&value);
    }
}

// Optional helpers so it's easy to get the prover state back out.
impl<R> SpongefishTranscript<R>
where
    R: RngCore + CryptoRng,
{
    pub fn new(prover_state: ProverState<StdHash, R>) -> Self {
        Self(prover_state)
    }
    pub fn into_inner(self) -> ProverState<StdHash, R> {
        self.0
    }
    pub fn as_inner(&self) -> &ProverState<StdHash, R> {
        &self.0
    }
    pub fn as_inner_mut(&mut self) -> &mut ProverState<StdHash, R> {
        &mut self.0
    }
}
