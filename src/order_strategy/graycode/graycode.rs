use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

pub struct GraycodeOrder {
    current_index: usize,
    stop_value: usize, // exclusive
    num_vars: usize,
}

impl GraycodeOrder {
    pub fn next_gray_code(value: usize) -> usize {
        let mask = match value.count_ones() & 1 == 0 {
            true => 1,
            false => 1 << (value.trailing_zeros() + 1),
        };
        value ^ mask
    }
}

impl OrderStrategy for GraycodeOrder {
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
            self.current_index = GraycodeOrder::next_gray_code(self.current_index);
            this_index
        } else {
            None
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}
