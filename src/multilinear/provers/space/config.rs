use ark_ff::Field;

use crate::{
    prover::{BatchProverConfig, ProverConfig},
    streams::Stream,
};

pub struct SpaceProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_variables: usize,
    pub claim: F,
    pub streams: Vec<S>,
}

impl<F, S> SpaceProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(claim: F, num_variables: usize, stream: S) -> Self {
        Self {
            claim,
            num_variables,
            streams: vec![stream],
        }
    }
}

impl<F: Field, S: Stream<F>> ProverConfig<F, S> for SpaceProverConfig<F, S> {
    fn default(claim: F, num_variables: usize, stream: S) -> Self {
        Self {
            claim,
            num_variables,
            streams: vec![stream],
        }
    }
}

impl<F: Field, S: Stream<F>> BatchProverConfig<F, S> for SpaceProverConfig<F, S> {
    fn default(claim: F, num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            claim,
            num_variables,
            streams,
        }
    }
}
