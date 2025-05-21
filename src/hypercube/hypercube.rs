use crate::{hypercube::HypercubeMember, order_strategy::OrderStrategy};

// On each call to next() this gives a HypercubeMember for the value
#[derive(Debug)]
pub struct Hypercube<O: OrderStrategy> {
    order: O,
}

impl<O: OrderStrategy> Hypercube<O> {
    pub fn new(num_vars: usize) -> Self {
        let order = O::new(num_vars);
        Self { order }
    }
    pub fn stop_value(num_vars: usize) -> usize {
        1 << num_vars // this is exclusive, meaning should stop *before* this value
    }
}

impl<O: OrderStrategy> Iterator for Hypercube<O> {
    type Item = (usize, HypercubeMember);
    fn next(&mut self) -> Option<Self::Item> {
        match self.order.next_index() {
            Some(current_index) => Some((
                current_index,
                HypercubeMember::new(self.order.num_vars(), current_index),
            )),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        hypercube::{Hypercube, HypercubeMember},
        order_strategy::{GraycodeOrder, LexicographicOrder},
    };

    fn is_eq(given: HypercubeMember, expected: Vec<bool>) {
        // check each value in the vec
        for (i, (a, b)) in given.zip(expected.clone()).enumerate() {
            assert_eq!(
                a, b,
                "bit at index {} incorrect, should be {:?}",
                i, expected
            );
        }
    }

    #[test]
    fn lexicographic_hypercube_members() {
        // for n=0, should return empty vec first call, none second call
        let mut hypercube_size_0 = Hypercube::<LexicographicOrder>::new(0);
        is_eq(hypercube_size_0.next().unwrap().1, vec![]);
        // for n=1, should return vec[false] first call, vec[true] second call and None third call
        let mut hypercube_size_1: Hypercube<LexicographicOrder> = Hypercube::new(1);
        is_eq(hypercube_size_1.next().unwrap().1, vec![false]);
        is_eq(hypercube_size_1.next().unwrap().1, vec![true]);
        assert_eq!(hypercube_size_1.next(), None);
        // so on for n=2
        let mut hypercube_size_2: Hypercube<LexicographicOrder> = Hypercube::new(2);
        is_eq(hypercube_size_2.next().unwrap().1, vec![false, false]);
        is_eq(hypercube_size_2.next().unwrap().1, vec![false, true]);
        is_eq(hypercube_size_2.next().unwrap().1, vec![true, false]);
        is_eq(hypercube_size_2.next().unwrap().1, vec![true, true]);
        assert_eq!(hypercube_size_2.next(), None);
        // so on for n=3
        let mut hypercube_size_3: Hypercube<LexicographicOrder> = Hypercube::new(3);
        is_eq(
            hypercube_size_3.next().unwrap().1,
            vec![false, false, false],
        );
        is_eq(hypercube_size_3.next().unwrap().1, vec![false, false, true]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![false, true, false]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![false, true, true]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, false, false]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, false, true]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, true, false]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, true, true]);
        assert_eq!(hypercube_size_3.next(), None);
    }

    #[test]
    fn lexicographic_indices() {
        // for n=0, should return empty vec first call, none second call
        let mut hypercube_size_0 = Hypercube::<LexicographicOrder>::new(0);
        assert_eq!(hypercube_size_0.next().unwrap().0, 0);
        // for n=1, should return vec[false] first call, vec[true] second call and None third call
        let mut hypercube_size_1: Hypercube<LexicographicOrder> = Hypercube::new(1);
        assert_eq!(hypercube_size_1.next().unwrap().0, 0);
        assert_eq!(hypercube_size_1.next().unwrap().0, 1);
        assert_eq!(hypercube_size_1.next(), None);
        // so on for n=2
        let mut hypercube_size_2: Hypercube<LexicographicOrder> = Hypercube::new(2);
        assert_eq!(hypercube_size_2.next().unwrap().0, 0);
        assert_eq!(hypercube_size_2.next().unwrap().0, 1);
        assert_eq!(hypercube_size_2.next().unwrap().0, 2);
        assert_eq!(hypercube_size_2.next().unwrap().0, 3);
        assert_eq!(hypercube_size_2.next(), None);
        // so on for n=3
        let mut hypercube_size_3: Hypercube<LexicographicOrder> = Hypercube::new(3);
        assert_eq!(hypercube_size_3.next().unwrap().0, 0);
        assert_eq!(hypercube_size_3.next().unwrap().0, 1);
        assert_eq!(hypercube_size_3.next().unwrap().0, 2);
        assert_eq!(hypercube_size_3.next().unwrap().0, 3);
        assert_eq!(hypercube_size_3.next().unwrap().0, 4);
        assert_eq!(hypercube_size_3.next().unwrap().0, 5);
        assert_eq!(hypercube_size_3.next().unwrap().0, 6);
        assert_eq!(hypercube_size_3.next().unwrap().0, 7);
        assert_eq!(hypercube_size_3.next(), None);
    }

    #[test]
    fn graycode_hypercube_members() {
        // https://docs.rs/gray-codes/latest/gray_codes/struct.GrayCode8.html#examples
        // for n=0, should return empty vec first call, none second call
        let mut hypercube_size_0 = Hypercube::<GraycodeOrder>::new(0);
        is_eq(hypercube_size_0.next().unwrap().1, vec![]);
        // for n=1, should return vec[false] first call, vec[true] second call and None third call
        let mut hypercube_size_1: Hypercube<GraycodeOrder> = Hypercube::new(1);
        is_eq(hypercube_size_1.next().unwrap().1, vec![false]);
        is_eq(hypercube_size_1.next().unwrap().1, vec![true]);
        assert_eq!(hypercube_size_1.next(), None);
        // so on for n=2
        let mut hypercube_size_2: Hypercube<GraycodeOrder> = Hypercube::new(2);
        is_eq(hypercube_size_2.next().unwrap().1, vec![false, false]);
        is_eq(hypercube_size_2.next().unwrap().1, vec![false, true]);
        is_eq(hypercube_size_2.next().unwrap().1, vec![true, true]);
        is_eq(hypercube_size_2.next().unwrap().1, vec![true, false]);
        assert_eq!(hypercube_size_2.next(), None);
        // so on for n=3
        let mut hypercube_size_3: Hypercube<GraycodeOrder> = Hypercube::new(3);
        is_eq(
            hypercube_size_3.next().unwrap().1,
            vec![false, false, false],
        );
        is_eq(hypercube_size_3.next().unwrap().1, vec![false, false, true]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![false, true, true]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![false, true, false]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, true, false]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, true, true]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, false, true]);
        is_eq(hypercube_size_3.next().unwrap().1, vec![true, false, false]);
        assert_eq!(hypercube_size_3.next(), None);
    }

    #[test]
    fn graycode_indices() {
        // https://docs.rs/gray-codes/latest/gray_codes/struct.GrayCode8.html#examples
        // for n=0, should return empty vec first call, none second call
        let mut hypercube_size_0 = Hypercube::<GraycodeOrder>::new(0);
        assert_eq!(hypercube_size_0.next().unwrap().0, 0);
        // for n=1, should return vec[false] first call, vec[true] second call and None third call
        let mut hypercube_size_1: Hypercube<GraycodeOrder> = Hypercube::new(1);
        assert_eq!(hypercube_size_1.next().unwrap().0, 0);
        assert_eq!(hypercube_size_1.next().unwrap().0, 1);
        assert_eq!(hypercube_size_1.next(), None);
        // so on for n=2
        let mut hypercube_size_2: Hypercube<GraycodeOrder> = Hypercube::new(2);
        assert_eq!(hypercube_size_2.next().unwrap().0, 0);
        assert_eq!(hypercube_size_2.next().unwrap().0, 1);
        assert_eq!(hypercube_size_2.next().unwrap().0, 3);
        assert_eq!(hypercube_size_2.next().unwrap().0, 2);
        assert_eq!(hypercube_size_2.next(), None);
        // so on for n=3
        let mut hypercube_size_3: Hypercube<GraycodeOrder> = Hypercube::new(3);
        assert_eq!(hypercube_size_3.next().unwrap().0, 0);
        assert_eq!(hypercube_size_3.next().unwrap().0, 1);
        assert_eq!(hypercube_size_3.next().unwrap().0, 3);
        assert_eq!(hypercube_size_3.next().unwrap().0, 2);
        assert_eq!(hypercube_size_3.next().unwrap().0, 6);
        assert_eq!(hypercube_size_3.next().unwrap().0, 7);
        assert_eq!(hypercube_size_3.next().unwrap().0, 5);
        assert_eq!(hypercube_size_3.next().unwrap().0, 4);
        assert_eq!(hypercube_size_3.next(), None);
    }
}
