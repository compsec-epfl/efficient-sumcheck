pub trait OrderStrategy {
    fn new(num_variables: usize) -> Self;
    fn next_index(&mut self) -> Option<usize>;
    fn num_vars(&self) -> usize;
}
