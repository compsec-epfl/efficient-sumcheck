pub mod pairwise;
pub mod variablewise;

#[derive(Copy, Clone, Debug)]
pub enum ReduceMode {
    Pairwise,
    Variablewise,
}
