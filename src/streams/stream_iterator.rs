use std::marker::PhantomData;

use crate::{order_strategy::OrderStrategy, streams::Stream};
use ark_ff::Field;

pub struct StreamIterator<F: Field, S: Stream<F>, O: OrderStrategy> {
    stream: S,
    order: O,
    _marker: PhantomData<F>,
}

impl<F: Field, S: Stream<F>, O: OrderStrategy> StreamIterator<F, S, O> {
    pub fn new(stream: S) -> Self {
        let order = O::new_from_num_vars(stream.num_variables());
        Self {
            stream,
            order,
            _marker: PhantomData,
        }
    }
    pub fn reset(&mut self) {
        self.order = O::new_from_num_vars(self.stream.num_variables());
    }
}

impl<F: Field, S: Stream<F>, O: OrderStrategy> Iterator for StreamIterator<F, S, O> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        match self.order.next_index() {
            Some(index) => Some(self.stream.evaluation(index)),
            None => None,
        }
    }
}
