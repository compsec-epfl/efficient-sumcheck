pub trait OrderStrategy: Iterator<Item = usize> {
    fn new(len: usize) -> Self;
    fn new_from_num_vars(num_variables: usize) -> Self;
    fn next_index(&mut self) -> Option<usize>;
    fn num_vars(&self) -> usize;
}
