use ark_ff::Field;

use crate::{prover::ProverConfig, streams::Stream};

pub struct SpaceProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_variables: usize,
    pub claim: F,
    pub stream: S,
}

impl<'a, F, S> SpaceProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(claim: F, num_variables: usize, stream: S) -> Self {
        Self {
            claim,
            num_variables,
            stream,
        }
    }
}

impl<F: Field, S: Stream<F>> ProverConfig<F, S> for SpaceProverConfig<F, S> {
    fn default(claim: F, num_variables: usize, stream: S) -> Self {
        Self {
            claim,
            num_variables,
            stream,
        }
    }
}
