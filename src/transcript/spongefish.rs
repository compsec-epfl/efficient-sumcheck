use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::{Decoding, Encoding, ProverState, StdHash};

use crate::transcript::Transcript;

/// Newtype wrapper around spongefish's [`ProverState`] so we can implement [`Transcript`].
///
/// Maps `send` → `prover_message`, `challenge` → `verifier_message`.
/// `receive` also uses `verifier_message` (for test/sanity use on the prover side);
/// a proper verifier transcript wrapping `VerifierState` would use `prover_message()?`.
pub struct SpongefishTranscript<R: RngCore + CryptoRng = ark_std::rand::rngs::StdRng>(
    pub ProverState<StdHash, R>,
);

impl<F, R> Transcript<F> for SpongefishTranscript<R>
where
    F: Field + Encoding<[u8]> + Decoding<[u8]>,
    R: RngCore + CryptoRng,
{
    type Error = core::convert::Infallible;

    fn send(&mut self, value: F) {
        self.0.prover_message(&value);
    }

    fn receive(&mut self) -> Result<F, Self::Error> {
        // On the prover side, "receive" doesn't make semantic sense —
        // this is here for trait completeness. Returns a challenge.
        Ok(self.0.verifier_message::<F>())
    }

    fn challenge(&mut self) -> F {
        self.0.verifier_message::<F>()
    }
}

/// Blanket impl so raw `ProverState` can be used as a `Transcript` directly.
impl<F, H, R> Transcript<F> for spongefish::ProverState<H, R>
where
    F: Field + Encoding<[H::U]> + Decoding<[H::U]> + spongefish::NargSerialize,
    H: spongefish::DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    type Error = core::convert::Infallible;

    fn send(&mut self, value: F) {
        self.prover_message(&value);
    }

    fn receive(&mut self) -> Result<F, Self::Error> {
        Ok(self.verifier_message::<F>())
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
