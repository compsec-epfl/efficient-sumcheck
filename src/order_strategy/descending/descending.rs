use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

#[derive(PartialEq, Debug)]
pub struct DescendingOrder {
    current_index: isize,
    stop_value: isize,
    num_vars: usize,
}

impl OrderStrategy for DescendingOrder {
    fn new(num_vars: usize) -> Self {
        Self {
            current_index: Hypercube::<Self>::stop_value(num_vars) as isize - 1,
            stop_value: 0,
            num_vars,
        }
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current_index < self.stop_value {
            return None;
        }

        let this_index = self.current_index as usize;
        self.current_index -= 1;
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
        let order_0 = DescendingOrder::new(0);
        let indices_0: Vec<usize> = order_0.collect();
        assert_eq!(indices_0, vec![0]);

        let order_1 = DescendingOrder::new(1);
        let indices_1: Vec<usize> = order_1.collect();
        assert_eq!(indices_1, vec![1, 0]);

        let order_2 = DescendingOrder::new(2);
        let indices_2: Vec<usize> = order_2.collect();
        assert_eq!(indices_2, vec![3, 2, 1, 0]);

        let order_3 = DescendingOrder::new(3);
        let indices_3: Vec<usize> = order_3.collect();
        assert_eq!(indices_3, vec![7, 6, 5, 4, 3, 2, 1, 0]);
    }
}