//! Concrete [`SumcheckProver`](crate::sumcheck_prover::SumcheckProver)
//! implementations for each polynomial shape.

#[cfg(feature = "arkworks")]
pub mod coefficient;
#[cfg(feature = "arkworks")]
pub mod coefficient_lsb;
pub mod eq_factored;
pub mod gkr;
pub mod inner_product;
pub mod inner_product_lsb;
pub mod multilinear;
pub mod multilinear_lsb;
