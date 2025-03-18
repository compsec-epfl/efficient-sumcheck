use crate::{hypercube::Hypercube, streams::order_strategy::OrderStrategy};

pub struct LexicographicOrder {
    current_index: usize,
    stop_value: usize, // exclusive
}

impl OrderStrategy for LexicographicOrder {
    fn new(num_variables: usize) -> Self {
        Self {
            current_index: 0,
            stop_value: Hypercube::stop_value(num_variables), // exclusive
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
}
