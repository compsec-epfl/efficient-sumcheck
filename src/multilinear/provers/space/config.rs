use std::marker::PhantomData;

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
    pub streams: Vec<S>,
    _f: PhantomData<F>,
}

impl<F, S> SpaceProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(num_variables: usize, stream: S) -> Self {
        Self {
            num_variables,
            streams: vec![stream],
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> ProverConfig<F, S> for SpaceProverConfig<F, S> {
    fn default(num_variables: usize, stream: S) -> Self {
        Self {
            num_variables,
            streams: vec![stream],
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> BatchProverConfig<F, S> for SpaceProverConfig<F, S> {
    fn default(num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            num_variables,
            streams,
            _f: PhantomData::<F>,
        }
    }
}
