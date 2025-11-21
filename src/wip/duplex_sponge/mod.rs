use ark_ff::{Field, UniformRand};
use ark_std::rand::{Rng, SeedableRng};
use core::marker::PhantomData;
use spongefish::DuplexSpongeInterface;
use zeroize::Zeroize;

#[derive(Clone, Zeroize)]
pub struct BenchDuplexSponge<F: Field, R: Rng + SeedableRng<Seed = [u8; 32]>> {
    #[zeroize(skip)] // skip zeroizing RNG state (optional)
    rng: R,
    _marker: PhantomData<F>,
}

impl<F: Field, R: Rng + SeedableRng<Seed = [u8; 32]>> Default for BenchDuplexSponge<F, R> {
    fn default() -> Self {
        // Fixed seed for reproducible benchmarks
        Self {
            rng: R::from_seed([0u8; 32]),
            _marker: PhantomData,
        }
    }
}

impl<F: Field + UniformRand, R: Rng + SeedableRng<Seed = [u8; 32]>> BenchDuplexSponge<F, R> {
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            _marker: PhantomData,
        }
    }
}

impl<F, R> DuplexSpongeInterface for BenchDuplexSponge<F, R>
where
    F: Field + UniformRand,
    R: Rng + SeedableRng<Seed = [u8; 32]> + Clone,
{
    fn new(iv: [u8; 32]) -> Self {
        // For a bench sponge, just seed the RNG from the IV
        Self {
            rng: R::from_seed(iv),
            _marker: PhantomData,
        }
    }

    fn absorb_unchecked(&mut self, _input: &[u8]) -> &mut Self {
        // bench stub: ignore input
        self
    }

    fn squeeze_unchecked(&mut self, output: &mut [u8]) -> &mut Self {
        // bench stub: just fill with rng output
        self.rng.fill(output);
        self
    }

    fn ratchet_unchecked(&mut self) -> &mut Self {
        // bench stub: maybe draw and discard some random bytes if you want
        self
    }
}
