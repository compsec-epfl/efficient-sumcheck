pub trait OrderStrategy {
    fn new(num_variables: usize) -> Self;
    fn next_index(&mut self) -> Option<usize>;
}
