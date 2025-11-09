use crate::order_strategy::OrderStrategy;

#[derive(Clone, Debug, PartialEq)]
pub struct HypercubeMember<O: OrderStrategy> {
    order: O,
    bit_index: usize,
    num_vars: usize,
    value: usize,
}

impl<O: OrderStrategy> HypercubeMember<O> {
    pub fn new(len: usize, value: usize) -> Self {
        assert!(len <= std::mem::size_of::<usize>() * 8);
        println!("new");
        Self {
            order: O::new(len),
            bit_index: len,
            num_vars: len,
            value,
        }
    }
    pub fn new_from_vec_bool(value: Vec<bool>) -> Self {
        HypercubeMember::new(
            value.len(),
            HypercubeMember::<O>::usize_from_vec_bool(value),
        )
    }
    pub fn len(&self) -> usize {
        self.num_vars
    }
    pub fn is_empty(&self) -> bool {
        if self.bit_index == 0 {
            return true;
        }
        false
    }
    pub fn usize_from_vec_bool(vec: Vec<bool>) -> usize {
        vec.into_iter()
            .rev()
            .enumerate()
            .fold(0, |acc, (i, bit)| acc | ((bit as usize) << i))
    }
    pub fn elements_at_indices(b: Vec<bool>, indices: Vec<usize>) -> Vec<bool> {
        // checks
        if indices.is_empty() {
            return vec![];
        }
        assert!(b.len() >= indices.len());
        assert!(b.len() > *indices.last().unwrap());
        // get the indices
        let mut b_prime: Vec<bool> = Vec::with_capacity(indices.len());
        for index in &indices {
            b_prime.push(b[*index]);
        }
        b_prime
    }
    pub fn to_vec_bool(&self) -> Vec<bool> {
        let mut b: Vec<bool> = Vec::with_capacity(self.num_vars);
        for bit_index in (0..self.num_vars).rev() {
            b.push(self.value & (1 << bit_index) != 0);
        }
        b
    }
    pub fn value(&self) -> usize {
        self.value
    }
}

impl<O: OrderStrategy> Iterator for HypercubeMember<O> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        // Check if n == 0
        if self.bit_index == 0 {
            return None;
        }
        // Return if value is bit high at bit_index
        self.bit_index -= 1;
        println!(
            "self.bit_index: {}, next(): {}",
            self.bit_index,
            self.order.next().unwrap()
        );
        let bit_mask = 1 << self.bit_index;
        Some(self.value & bit_mask != 0)
    }
}

#[cfg(test)]
mod tests {
    use crate::hypercube::HypercubeMember;
    use crate::order_strategy::AscendingOrder;

    #[test]
    fn sanity() {
        let test_0_0 = HypercubeMember::<AscendingOrder>::new(0, 0);
        let vals_0_0: Vec<bool> = test_0_0.collect();
        assert_eq!(vals_0_0, vec![]);

        let test_0_1 = HypercubeMember::<AscendingOrder>::new(0, 1);
        let vals_0_1: Vec<bool> = test_0_1.collect();
        assert_eq!(vals_0_1, vec![]);

        let test_1_0 = HypercubeMember::<AscendingOrder>::new(1, 0);
        let vals_1_0: Vec<bool> = test_1_0.collect();
        assert_eq!(vals_1_0, vec![false]);

        let test_1_1 = HypercubeMember::<AscendingOrder>::new(1, 1);
        let vals_1_1: Vec<bool> = test_1_1.collect();
        assert_eq!(vals_1_1, vec![true]);

        let test_2_0 = HypercubeMember::<AscendingOrder>::new(2, 0);
        let vals_2_0: Vec<bool> = test_2_0.collect();
        assert_eq!(vals_2_0, vec![false, false]);

        let test_2_1 = HypercubeMember::<AscendingOrder>::new(2, 1);
        let vals_2_1: Vec<bool> = test_2_1.collect();
        assert_eq!(vals_2_1, vec![false, true]);

        let test_2_2 = HypercubeMember::<AscendingOrder>::new(2, 2);
        let vals_2_2: Vec<bool> = test_2_2.collect();
        assert_eq!(vals_2_2, vec![true, false]);

        let test_2_3 = HypercubeMember::<AscendingOrder>::new(2, 3);
        let vals_2_3: Vec<bool> = test_2_3.collect();
        assert_eq!(vals_2_3, vec![true, true]);

        let test_3_3 = HypercubeMember::<AscendingOrder>::new(3, 0);
        let vals_3_3: Vec<bool> = test_3_3.collect();
        assert_eq!(vals_3_3, vec![false, false, false]);
    }

    #[test]
    fn elements_at_indices() {
        let test_1 = vec![true, false, false, false, false];
        let indices_1 = vec![2, 3];
        let result_1 = HypercubeMember::<AscendingOrder>::elements_at_indices(test_1, indices_1);
        assert_eq!(result_1, vec![false, false]);
        let test_2 = vec![false, true, true, false, false, false, false, true];
        let indices_2 = vec![0, 1, 2, 4, 6];
        let result_2 = HypercubeMember::<AscendingOrder>::elements_at_indices(test_2, indices_2);
        assert_eq!(result_2, vec![false, true, true, false, false]);
    }
    #[test]
    fn vec_bool_to_usize() {
        let test_1 = vec![true, false, false];
        let exp_1 = 4;
        assert_eq!(
            HypercubeMember::<AscendingOrder>::usize_from_vec_bool(test_1),
            exp_1
        );
        let test_2 = vec![false, true, true];
        let exp_2 = 3;
        assert_eq!(
            HypercubeMember::<AscendingOrder>::usize_from_vec_bool(test_2),
            exp_2
        );
    }
    #[test]
    fn to_vec_bool() {
        let exp_1 = vec![true, false, false, false, false];
        let test_1 = HypercubeMember::<AscendingOrder>::new_from_vec_bool(exp_1.clone());
        assert_eq!(exp_1, test_1.to_vec_bool());
        let test_2 = HypercubeMember::<AscendingOrder>::new(5, 16);
        assert_eq!(exp_1, test_2.to_vec_bool());

        let exp_2 = vec![false, false, true, false, true];
        let test_3 = HypercubeMember::<AscendingOrder>::new_from_vec_bool(exp_2.clone());
        assert_eq!(exp_2, test_3.to_vec_bool());
        let test_4 = HypercubeMember::<AscendingOrder>::new(5, 5);
        assert_eq!(exp_2, test_4.to_vec_bool());

        let exp_3 = vec![false, false, true];
        let test_3 = HypercubeMember::<AscendingOrder>::new(3, 1);
        assert_eq!(test_3.to_vec_bool(), exp_3);
    }
}
