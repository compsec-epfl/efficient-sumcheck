use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::{Decoding, Encoding, ProverState, StdHash};

use crate::transcript::ProverTranscript;

/// Spongefish prover transcript.
///
/// Implements [`ProverTranscript`] only — the verifier side should wrap
/// spongefish's `VerifierState` and implement [`VerifierTranscript`](super::VerifierTranscript).
pub struct SpongefishTranscript<R: RngCore + CryptoRng = ark_std::rand::rngs::StdRng>(
    pub ProverState<StdHash, R>,
);

impl<F, R> ProverTranscript<F> for SpongefishTranscript<R>
where
    F: Field + Encoding<[u8]> + Decoding<[u8]>,
    R: RngCore + CryptoRng,
{
    fn send(&mut self, value: F) {
        self.0.prover_message(&value);
    }

    fn challenge(&mut self) -> F {
        self.0.verifier_message::<F>()
    }
}

/// Blanket impl so raw `ProverState` can be used as a `ProverTranscript` directly.
impl<F, H, R> ProverTranscript<F> for spongefish::ProverState<H, R>
where
    F: Field + Encoding<[H::U]> + Decoding<[H::U]> + spongefish::NargSerialize,
    H: spongefish::DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    fn send(&mut self, value: F) {
        self.prover_message(&value);
    }

    fn challenge(&mut self) -> F {
        self.verifier_message::<F>()
    }
}

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
