use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

pub struct LexicographicOrder {
    current_index: usize,
    stop_value: usize, // exclusive
    num_vars: usize,
}

impl OrderStrategy for LexicographicOrder {
    fn new(num_vars: usize) -> Self {
        Self {
            current_index: 0,
            stop_value: Hypercube::<Self>::stop_value(num_vars), // exclusive
            num_vars,
        }
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current_index < self.stop_value {
            let this_index = Some(self.current_index);
            self.current_index += 1;
            this_index
        } else {
            None
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl Iterator for LexicographicOrder {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_index()
    }
}
