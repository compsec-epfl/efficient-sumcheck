use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

pub struct SignificantBitOrder {
    current_index: usize,
    stop_value: usize, // exclusive
    num_vars: usize,
}

// we're using the usize like a vec<bool>, so we can't just reverse the whole thing .reverse_bits()
fn reverse_lsb(x: usize, n: u32) -> usize {
    let mut result = 0;
    for i in 0..n {
        let bit = (x >> i) & 1;
        result |= bit << (n - 1 - i);
    }
    result
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
            let this_index = Some(reverse_lsb(self.current_index, self.num_vars as u32));
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

impl Iterator for SignificantBitOrder {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_index()
    }
}
