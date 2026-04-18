/// Prover-side Fiat-Shamir transcript.
///
/// The prover absorbs messages into the sponge and squeezes challenges.
/// No `receive` — the prover never reads prover messages.
///
/// ```ignore
/// transcript.send(c0);            // absorb prover message
/// transcript.send(c2);            // absorb prover message
/// let r = transcript.challenge(); // squeeze challenge
/// ```
pub trait ProverTranscript<T> {
    /// Absorb a prover message into the transcript.
    fn send(&mut self, value: T);

    /// Squeeze a verifier challenge from the transcript.
    fn challenge(&mut self) -> T;
}

/// Verifier-side Fiat-Shamir transcript.
///
/// The verifier reads (absorbs + decodes) prover messages and squeezes
/// the same challenges as the prover. No `send` — the verifier never
/// produces prover messages.
///
/// ```ignore
/// let c0 = transcript.receive()?; // absorb + decode prover message
/// let c2 = transcript.receive()?; // absorb + decode prover message
/// let r = transcript.challenge();   // squeeze challenge (same as prover)
/// ```
pub trait VerifierTranscript<T> {
    type Error: core::fmt::Debug;

    /// Read a prover message from the transcript.
    ///
    /// Absorbs + decodes the next prover message. Returns `Err` if the
    /// transcript data is malformed or exhausted.
    fn receive(&mut self) -> Result<T, Self::Error>;

    /// Squeeze a verifier challenge from the transcript.
    ///
    /// Deterministic given the absorbed state — same on prover and verifier.
    fn challenge(&mut self) -> T;
}
