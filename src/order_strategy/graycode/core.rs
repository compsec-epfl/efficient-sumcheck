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

impl Iterator for GraycodeOrder {
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
        let order_0 = GraycodeOrder::new(0);
        let indices_0: Vec<usize> = order_0.collect();
        assert_eq!(indices_0, vec![0]);

        let order_1 = GraycodeOrder::new(1);
        let indices_1: Vec<usize> = order_1.collect();
        assert_eq!(indices_1, vec![0, 1]);

        let order_2 = GraycodeOrder::new(2);
        let indices_2: Vec<usize> = order_2.collect();
        assert_eq!(indices_2, vec![0, 1, 3, 2]);

        let order_3 = GraycodeOrder::new(3);
        let indices_3: Vec<usize> = order_3.collect();
        assert_eq!(indices_3, vec![0, 1, 3, 2, 6, 7, 5, 4]);
    }
}
