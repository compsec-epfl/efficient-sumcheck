use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

pub struct SignificantBitOrder {
    current_index: usize,
    stop_value: usize, // exclusive
    num_vars: usize,
}

// we're using the usize like a vec<bool>, so we can't just reverse the whole thing .reverse_bits()
impl SignificantBitOrder {
    pub fn next_value_in_msb_order(x: usize, n: u32) -> usize {
        let mut result = x;
        for i in (0..n).rev() {
            result ^= 1 << i;
            if result >> i == 1 {
                break;
            }
        }
        result
    }
}

impl OrderStrategy for SignificantBitOrder {
    fn new(num_vars: usize) -> Self {
        Self {
            current_index: 0,
            stop_value: Hypercube::<Self>::stop_value(num_vars), // exclusive
            num_vars,
        }
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current_index < self.stop_value {
            let old_index = self.current_index;
            self.current_index = SignificantBitOrder::next_value_in_msb_order(
                self.current_index,
                self.num_vars as u32,
            );
            if self.current_index == 0 {
                // if the sequence rounds back to 0, we need to stop
                self.current_index = self.stop_value;
            }
            Some(old_index)
        } else {
            None
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl Iterator for SignificantBitOrder {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_index()
    }
}
