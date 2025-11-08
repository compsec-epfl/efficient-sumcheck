use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

pub struct MSBOrder {
    current_index: usize,
    stop_value: usize, // exclusive
    num_vars: usize,
}

// we're using the usize like a vec<bool>, so we can't just reverse the whole thing .reverse_bits()
impl MSBOrder {
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

impl OrderStrategy for MSBOrder {
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
            self.current_index =
                MSBOrder::next_value_in_msb_order(self.current_index, self.num_vars as u32);
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

impl Iterator for MSBOrder {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity() {
        // https://docs.rs/gray-codes/latest/gray_codes/struct.GrayCode8.html#examples
        let order_0 = MSBOrder::new(0);
        let indices_0: Vec<usize> = order_0.collect();
        assert_eq!(indices_0, vec![0]);

        let order_1 = MSBOrder::new(1);
        let indices_1: Vec<usize> = order_1.collect();
        assert_eq!(indices_1, vec![0, 1]);

        let order_2 = MSBOrder::new(2);
        let indices_2: Vec<usize> = order_2.collect();
        assert_eq!(indices_2, vec![0, 2, 1, 3]);

        let order_3 = MSBOrder::new(3);
        let indices_3: Vec<usize> = order_3.collect();
        assert_eq!(indices_3, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }
}
