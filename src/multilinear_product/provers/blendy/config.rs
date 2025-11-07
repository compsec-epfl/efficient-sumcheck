use std::marker::PhantomData;

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
    pub streams: Vec<S>,
    _f: PhantomData<F>,
}

impl<F, S> BlendyProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(num_stages: usize, num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            num_stages,
            num_variables,
            streams,
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> ProductProverConfig<F, S> for BlendyProductProverConfig<F, S> {
    fn default(num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            num_stages: DEFAULT_NUM_STAGES,
            num_variables,
            streams,
            _f: PhantomData::<F>,
        }
    }
}
