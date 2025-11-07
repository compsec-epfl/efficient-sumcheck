pub mod pairwise;
pub mod tablewise;
pub mod variablewise;

#[derive(Copy, Clone, Debug)]
pub enum ReduceMode {
    Pairwise,
    Variablewise,
}
