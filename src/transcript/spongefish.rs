//! Bridge impls between effsc's transcript traits and spongefish.
//!
//! Lives here (rather than in a separate `effsc-spongefish` crate) because
//! the orphan rule forces the impl into a crate that owns one of the two
//! traits, and a separate crate would only add Cargo overhead. Disable the
//! `spongefish` feature to compile effsc without this module.

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::{
    Decoding, DuplexSpongeInterface, Encoding, NargDeserialize, NargSerialize, ProverState,
    VerifierState,
};

use crate::transcript::{ProverTranscript, VerifierTranscript};

impl<F, H, R> ProverTranscript<F> for ProverState<H, R>
where
    F: Field + Encoding<[H::U]> + Decoding<[H::U]> + NargSerialize,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    fn send(&mut self, value: F) {
        self.prover_message(&value);
    }
    fn challenge(&mut self) -> F {
        self.verifier_message::<F>()
    }
}

impl<'a, F, H> VerifierTranscript<F> for VerifierState<'a, H>
where
    F: Field + Encoding<[H::U]> + Decoding<[H::U]> + NargDeserialize,
    H: DuplexSpongeInterface,
{
    type Error = spongefish::VerificationError;

    fn receive(&mut self) -> Result<F, Self::Error> {
        self.prover_message::<F>()
    }
    fn challenge(&mut self) -> F {
        self.verifier_message::<F>()
    }
}
