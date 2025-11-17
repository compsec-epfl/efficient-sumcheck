use ark_ff::{Field, UniformRand};
use ark_std::rand::Rng;
use core::marker::PhantomData;

use crate::wip::fiat_shamir::fiat_shamir::FiatShamir;

pub struct BenchFiatShamir<F: Field, R: Rng> {
    rng: R,
    _marker: PhantomData<F>,
}

impl<F: Field, R: Rng> BenchFiatShamir<F, R> {
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, R: Rng> FiatShamir<F> for BenchFiatShamir<F, R>
where
    F: Field + UniformRand,
    R: Rng,
{
    fn absorb(&mut self, _value: F) {
        // Intentionally No-Op
    }

    fn squeeze(&mut self) -> F {
        F::rand(&mut self.rng)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::SmallM31,
        wip::fiat_shamir::{bench::BenchFiatShamir, FiatShamir},
    };

    #[test]
    fn test_bench_fiat_shamir() {
        let mut fs = BenchFiatShamir::<SmallM31, _>::new(ark_std::test_rng());
        fs.absorb(SmallM31::from(42u64));
        let challenge: SmallM31 = fs.squeeze();
        assert!(challenge != SmallM31::from(0));
    }
}
