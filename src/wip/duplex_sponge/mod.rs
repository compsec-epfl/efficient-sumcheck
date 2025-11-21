// use ark_ff::{Field, UniformRand};
// use ark_std::rand::Rng;
// use core::marker::PhantomData;
// use spongefish::{DuplexSpongeInterface, Unit};
// use zeroize::Zeroize;

// #[derive(Clone, Zeroize)]
// pub struct BenchDuplexSponge<F: Field, R: Rng> {
//     rng: R,
//     _marker: PhantomData<F>,
// }

// impl<F: Field, R: Rng> BenchDuplexSponge<F, R> {
//     pub fn new(rng: R) -> Self {
//         Self {
//             rng,
//             _marker: PhantomData,
//         }
//     }
// }

// impl<F, R> DuplexSpongeInterface for BenchDuplexSponge<F, R>
// where
//     F: Field + UniformRand,
//     R: Rng + Clone,
// {
//     fn new(iv: [u8; 32]) -> Self {
//         todo!()
//     }

//     fn absorb_unchecked(&mut self, input: &[u8]) -> &mut Self {
//         todo!()
//     }

//     fn squeeze_unchecked(&mut self, output: &mut [u8]) -> &mut Self {
//         todo!()
//     }

//     fn ratchet_unchecked(&mut self) -> &mut Self {
//         todo!()
//     }
// }
