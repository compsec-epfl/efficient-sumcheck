//! Concrete [`SumcheckProver`](crate::sumcheck_prover::SumcheckProver)
//! implementations for each polynomial shape.

#[cfg(feature = "arkworks")]
pub mod coefficient;
#[cfg(feature = "arkworks")]
pub mod coefficient_lsb;
#[cfg(feature = "arkworks")]
pub mod inner_product;
pub mod inner_product_lsb;
#[cfg(feature = "arkworks")]
pub mod multilinear;
pub mod multilinear_lsb;
