use crate::{hypercube::Hypercube, order_strategy::OrderStrategy};

pub struct AscendingOrder {
    current_index: usize,
    stop_value: usize, // exclusive
    num_vars: usize,
}

impl OrderStrategy for AscendingOrder {
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

impl Iterator for AscendingOrder {
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
        let order_0 = AscendingOrder::new(0);
        let indices_0: Vec<usize> = order_0.collect();
        assert_eq!(indices_0, vec![0]);

        let order_1 = AscendingOrder::new(1);
        let indices_1: Vec<usize> = order_1.collect();
        assert_eq!(indices_1, vec![0, 1]);

        let order_2 = AscendingOrder::new(2);
        let indices_2: Vec<usize> = order_2.collect();
        assert_eq!(indices_2, vec![0, 1, 2, 3]);

        let order_3 = AscendingOrder::new(3);
        let indices_3: Vec<usize> = order_3.collect();
        assert_eq!(indices_3, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }
}