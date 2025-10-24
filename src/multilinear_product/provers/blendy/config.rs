use ark_ff::Field;

use crate::{prover::ProductProverConfig, streams::Stream};

const DEFAULT_NUM_STAGES: usize = 2;

pub struct BlendyProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_stages: usize,
    pub num_variables: usize,
    pub claim: F,
    pub streams: Vec<S>,
}

impl<F, S> BlendyProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(claim: F, num_stages: usize, num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            claim,
            num_stages,
            num_variables,
            streams,
        }
    }
}

impl<F: Field, S: Stream<F>> ProductProverConfig<F, S> for BlendyProductProverConfig<F, S> {
    fn default(claim: F, num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            claim,
            num_stages: DEFAULT_NUM_STAGES,
            num_variables,
            streams,
        }
    }
}
