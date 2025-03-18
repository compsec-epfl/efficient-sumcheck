use std::marker::PhantomData;

use crate::streams::{order_strategy::OrderStrategy, Stream};
use ark_ff::Field;

pub struct StreamIterator<F: Field, S: Stream<F>, O: OrderStrategy> {
    stream: S,
    order: O,
    _marker: PhantomData<F>,
}

impl<'a, F: Field, S: Stream<F>, O: OrderStrategy> StreamIterator<F, S, O> {
    pub fn new(stream: S) -> Self {
        let order = O::new(stream.num_variables());
        Self {
            stream,
            order,
            _marker: PhantomData,
        }
    }
}

impl<'a, F: Field, S: Stream<F>, O: OrderStrategy> Iterator for StreamIterator<F, S, O> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        match self.order.next_index() {
            Some(index) => Some(self.stream.evaluation(index)),
            None => None,
        }
    }
}
