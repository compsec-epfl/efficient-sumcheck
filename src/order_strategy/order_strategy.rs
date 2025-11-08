pub trait OrderStrategy: Iterator<Item = usize> + Clone {
    fn new(len: usize) -> Self;
    fn next_index(&mut self) -> Option<usize>;
    fn num_vars(&self) -> usize;
}
