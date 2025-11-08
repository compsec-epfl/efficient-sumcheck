#[allow(clippy::module_inception)]
mod order_strategy;

mod ascending;
mod descending;
mod graycode;
mod msb;

pub use ascending::AscendingOrder;
pub use order_strategy::OrderStrategy;
pub use descending::DescendingOrder;
pub use graycode::GraycodeOrder;
pub use msb::MSBOrder;
