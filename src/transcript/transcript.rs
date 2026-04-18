/// Fiat-Shamir transcript for the sumcheck protocol.
///
/// Separates prover messages (absorbed into the sponge) from verifier
/// challenges (squeezed from the sponge). This distinction matters for
/// correct spongefish interop — `prover_message` and `verifier_message`
/// are different sponge operations.
///
/// # Prover usage
///
/// ```ignore
/// transcript.send(c0);          // absorb prover message
/// transcript.send(c2);          // absorb prover message
/// let r = transcript.challenge(); // squeeze challenge
/// ```
///
/// # Verifier usage
///
/// ```ignore
/// let c0 = transcript.receive()?; // absorb + decode prover message
/// let c2 = transcript.receive()?; // absorb + decode prover message
/// let r = transcript.challenge();   // squeeze challenge (same as prover)
/// ```
pub trait Transcript<T> {
    type Error: core::fmt::Debug;

    /// Absorb a prover message into the transcript (prover side).
    fn send(&mut self, value: T);

    /// Read a prover message from the transcript (verifier side).
    ///
    /// Returns `Err` if the transcript data is malformed or exhausted.
    fn receive(&mut self) -> Result<T, Self::Error>;

    /// Squeeze a verifier challenge from the transcript.
    ///
    /// Deterministic given the absorbed state — same on prover and verifier.
    fn challenge(&mut self) -> T;
}
