use ark_ff::{Field, UniformRand};
use ark_std::rand::{Rng, SeedableRng};
use core::marker::PhantomData;
use spongefish::DuplexSpongeInterface;
use zeroize::Zeroize;

#[derive(Clone, Zeroize)]
pub struct TestDuplexSponge<F: Field, R: Rng + SeedableRng<Seed = [u8; 32]>> {
    #[zeroize(skip)]
    rng: R,
    _marker: PhantomData<F>,
}

impl<F: Field, R: Rng + SeedableRng<Seed = [u8; 32]>> Default for TestDuplexSponge<F, R> {
    fn default() -> Self {
        Self {
            rng: R::from_seed([0u8; 32]),
            _marker: PhantomData,
        }
    }
}

impl<F: Field + UniformRand, R: Rng + SeedableRng<Seed = [u8; 32]>> TestDuplexSponge<F, R> {
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            _marker: PhantomData,
        }
    }
}

impl<F, R> DuplexSpongeInterface for TestDuplexSponge<F, R>
where
    F: Field + UniformRand,
    R: Rng + SeedableRng<Seed = [u8; 32]> + Clone,
{
    fn new(iv: [u8; 32]) -> Self {
        Self {
            rng: R::from_seed(iv),
            _marker: PhantomData,
        }
    }

    fn absorb_unchecked(&mut self, _input: &[u8]) -> &mut Self {
        self
    }

    fn squeeze_unchecked(&mut self, output: &mut [u8]) -> &mut Self {
        self.rng.fill(output);
        self
    }

    fn ratchet_unchecked(&mut self) -> &mut Self {
        self
    }
}
