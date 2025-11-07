use std::marker::PhantomData;

use ark_ff::Field;

use crate::{
    multilinear::provers::time::reductions::ReduceMode,
    prover::{BatchProverConfig, ProverConfig},
    streams::Stream,
};

pub struct TimeProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_variables: usize,
    pub streams: Vec<S>,
    pub reduce_mode: ReduceMode,
    _f: PhantomData<F>,
}

impl<F, S> TimeProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(num_variables: usize, stream: S, reduce_mode: ReduceMode) -> Self {
        Self {
            num_variables,
            streams: vec![stream],
            reduce_mode,
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> ProverConfig<F, S> for TimeProverConfig<F, S> {
    fn default(num_variables: usize, stream: S) -> Self {
        Self {
            num_variables,
            streams: vec![stream],
            reduce_mode: ReduceMode::Pairwise,
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> BatchProverConfig<F, S> for TimeProverConfig<F, S> {
    fn default(num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            num_variables,
            streams,
            reduce_mode: ReduceMode::Pairwise,
            _f: PhantomData::<F>,
        }
    }
}
