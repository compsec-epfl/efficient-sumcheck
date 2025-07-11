use ark_ff::Field;

use crate::{prover::ProverConfig, streams::Stream};

pub struct BlendyProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_stages: usize,
    pub num_variables: usize,
    pub claim: F,
    pub stream: S,
}

impl<'a, F, S> BlendyProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(claim: F, num_stages: usize, num_variables: usize, stream: S) -> Self {
        Self {
            claim,
            num_stages,
            num_variables,
            stream,
        }
    }
}

impl<F: Field, S: Stream<F>> ProverConfig<F, S> for BlendyProverConfig<F, S> {
    fn default(claim: F, num_variables: usize, stream: S) -> Self {
        Self {
            claim,
            num_stages: 2, // DEFAULT
            num_variables,
            stream,
        }
    }
}
