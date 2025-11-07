use std::marker::PhantomData;

use ark_ff::Field;

use crate::{prover::ProductProverConfig, streams::Stream};

pub struct SpaceProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_variables: usize,
    pub streams: Vec<S>,
    _f: PhantomData<F>,
}

impl<F, S> SpaceProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            num_variables,
            streams,
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> ProductProverConfig<F, S> for SpaceProductProverConfig<F, S> {
    fn default(num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            num_variables,
            streams,
            _f: PhantomData::<F>,
        }
    }
}
