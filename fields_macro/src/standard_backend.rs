use super::*;

pub fn standard_backend_impl(
    ty: proc_macro2::TokenStream,
    modulus: u128,
    generator: u128,
    suffix: &str,
) -> proc_macro2::TokenStream {
    // Type u128 has two limbs all other have only one
    let (from_bigint_impl, into_bigint_impl) = if suffix == "u128" {
        (
            quote! {
                fn from_bigint(a: BigInt<2>) -> Option<SmallFp<Self>> {
                    let val = (a.0[0] as u128) + ((a.0[0] as u128) << 64);
                    Some(SmallFp::new(val as Self::T))
                }
            },
            quote! {
                fn into_bigint(a: SmallFp<Self>) -> BigInt<2> {
                    ark_ff::BigInt([(a.value as u64), (a.value >> 64) as u64])
                }
            },
        )
    } else {
        (
            quote! {
                fn from_bigint(a: BigInt<2>) -> Option<SmallFp<Self>> {
                    if a.0[0] >= (Self::MODULUS as u64) {
                        None
                    } else {
                        Some(SmallFp::new(a.0[0] as Self::T))
                    }
                }
            },
            quote! {
                fn into_bigint(a: SmallFp<Self>) -> BigInt<2> {
                    ark_ff::BigInt([0, a.value as u64])
                }
            },
        )
    };

    quote! { type T = #ty;

        const MODULUS: Self::T = #modulus as Self::T;
        const MODULUS_128: u128 = #modulus;


        const GENERATOR: SmallFp<Self> = SmallFp::new(#generator as Self::T);
        const ZERO: SmallFp<Self> = SmallFp::new(0 as Self::T);
        const ONE: SmallFp<Self> = SmallFp::new(1 as Self::T);
        const TWO_ADIC_ROOT_OF_UNITY: SmallFp<Self> = SmallFp::new(1 as Self::T);

        const TWO_ADICITY: u32 = 0;
        const SQRT_PRECOMP: Option<SqrtPrecomputation<SmallFp<Self>>> = None;

        fn add_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            a.value = (a.value + b.value) % Self::MODULUS;
        }

        fn sub_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            if a.value >= b.value {
                a.value -= b.value;
            } else {
                a.value = Self::MODULUS - (b.value - a.value);
            }
        }

        fn double_in_place(a: &mut SmallFp<Self>) {
            a.value = (a.value + a.value) % Self::MODULUS;
        }

        fn neg_in_place(a: &mut SmallFp<Self>) {
            if a.value == (0 as Self::T) {
                a.value = 0 as Self::T;
            } else {
                a.value = Self::MODULUS - a.value;
            }
        }

        fn mul_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            let product = (a.value as u128) * (b.value as u128);
            a.value = (product % (Self::MODULUS as u128)) as Self::T;
        }

        fn sum_of_products<const T: usize>(
            a: &[SmallFp<Self>; T],
            b: &[SmallFp<Self>; T],) -> SmallFp<Self> {
            a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
        }

        fn square_in_place(a: &mut SmallFp<Self>) {
             let product = (a.value as u128) * (a.value as u128);
             a.value = (product % (Self::MODULUS as u128)) as Self::T;
        }

        // TODO: Do the EE algorithm in #ty
        fn inverse(a: &SmallFp<Self>) -> Option<SmallFp<Self>> {
            if a.value == 0 {
                return None;
            }

            let mut base: #ty = a.value;
            let mut exp: u128 = (Self::MODULUS_128 - 2);
            let mut acc: u128 = 1;
            let m: u128 = Self::MODULUS_128;

            while exp > 0 {
                if (exp & 1) == 1 {
                    acc = (acc * (base as u128)) % m;
                }
                base = (((base as u128) * (base as u128)) % m) as #ty;
                exp >>= 1;
            }
            Some(SmallFp::new(acc as Self::T))
        }

        #from_bigint_impl

        #into_bigint_impl
    }
}
