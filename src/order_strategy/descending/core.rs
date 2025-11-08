use ark_std::log2;

use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

#[derive(PartialEq, Debug)]
pub struct DescendingOrder {
    current_index: isize,
    stop_value_inclusive: isize,
    num_vars: usize,
}

impl OrderStrategy for DescendingOrder {
    fn new(len: usize) -> Self {
        let num_vars = match len {
            0 => 0,
            _ => log2(len) as usize,
        };
        println!("len: {:?}, num_vars: {:?}", len, num_vars);
        Self {
            current_index: len as isize - 1,
            stop_value_inclusive: 0,
            num_vars,
        }
    }

    fn new_from_num_vars(num_vars: usize) -> Self {
        let len = match num_vars {
            0 => 0,
            _ => Hypercube::<Self, Self>::stop_value(num_vars),
        };
        Self::new(len)
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current_index < self.stop_value_inclusive {
            return None;
        }

        let this_index = self.current_index as usize;
        self.current_index -= 1; // move downward: max, max-1, ..., 0
        Some(this_index)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl Iterator for DescendingOrder {
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
        let order_0 = DescendingOrder::new_from_num_vars(0);
        let indices_0: Vec<usize> = order_0.collect();
        assert_eq!(indices_0, vec![]);

        let order_1 = DescendingOrder::new_from_num_vars(1);
        let indices_1: Vec<usize> = order_1.collect();
        assert_eq!(indices_1, vec![1, 0]);

        let order_2 = DescendingOrder::new_from_num_vars(2);
        let indices_2: Vec<usize> = order_2.collect();
        assert_eq!(indices_2, vec![3, 2, 1, 0]);

        let order_3 = DescendingOrder::new_from_num_vars(3);
        let indices_3: Vec<usize> = order_3.collect();
        assert_eq!(indices_3, vec![7, 6, 5, 4, 3, 2, 1, 0]);
    }
}
