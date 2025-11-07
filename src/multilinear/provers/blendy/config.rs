use ark_ff::Field;
use ark_std::marker::PhantomData;

use crate::{prover::ProverConfig, streams::Stream};

pub struct BlendyProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_stages: usize,
    pub num_variables: usize,
    pub stream: S,
    _f: PhantomData<F>,
}

impl<F, S> BlendyProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(num_stages: usize, num_variables: usize, stream: S) -> Self {
        Self {
            num_stages,
            num_variables,
            stream,
            _f: PhantomData::<F>,
        }
    }
}

impl<F: Field, S: Stream<F>> ProverConfig<F, S> for BlendyProverConfig<F, S> {
    fn default(num_variables: usize, stream: S) -> Self {
        Self {
            num_stages: 2, // DEFAULT
            num_variables,
            stream,
            _f: PhantomData::<F>,
        }
    }
}
