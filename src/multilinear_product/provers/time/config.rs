use std::marker::PhantomData;

use ark_ff::Field;

use crate::{multilinear::ReduceMode, prover::ProductProverConfig, streams::Stream};

pub struct TimeProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_variables: usize,
    pub streams: Vec<S>,
    pub reduce_mode: ReduceMode,
    _f: PhantomData<F>,
}

impl<F, S> TimeProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(num_variables: usize, streams: Vec<S>, reduce_mode: ReduceMode) -> Self {
        Self {
            num_variables,
            streams,
            reduce_mode,
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> ProductProverConfig<F, S> for TimeProductProverConfig<F, S> {
    fn default(num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            num_variables,
            streams,
            reduce_mode: ReduceMode::Variablewise,
            _f: PhantomData::<F>,
        }
    }
}
