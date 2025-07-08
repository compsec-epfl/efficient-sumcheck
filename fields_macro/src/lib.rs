use core::convert::Into;
use proc_macro::TokenStream;
use quote::quote;
// use space_efficeint_sumcheck::fields::small_fp_backend::SmallFp;

use syn::{Expr, ExprLit, Lit, Meta};

extern crate proc_macro;
pub(crate) mod utils;

/// Fetch an attribute string from the derived struct.
fn fetch_attr(name: &str, attrs: &[syn::Attribute]) -> Option<String> {
    // Go over each attribute
    for attr in attrs {
        match attr.meta {
            // If the attribute's path matches `name`, and if the attribute is of
            // the form `#[name = "value"]`, return `value`
            Meta::NameValue(ref nv) if nv.path.is_ident(name) => {
                // Extract and return the string value.
                // If `value` is not a string, return an error
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(ref s),
                    ..
                }) = nv.value
                {
                    return Some(s.value());
                }
                panic!("attribute {name} should be a string")
            }
            _ => {}
        }
    }
    None
}

#[proc_macro_derive(SmallFpConfig, attributes(modulus, generator))]
pub fn fp_config(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let modulus: u128 = fetch_attr("modulus", &ast.attrs)
        .expect("Please supply a modulus attribute")
        .parse()
        .expect("Modulus should be a number");

    let generator: u128 = fetch_attr("generator", &ast.attrs)
        .expect("Please supply a generator attribute")
        .parse()
        .expect("Generator should be a number");

    let (ty, suffix) = {
        let u8_max = u128::from(u8::MAX);
        let u16_max = u128::from(u16::MAX);
        let u32_max = u128::from(u32::MAX);
        let u64_max = u128::from(u64::MAX);

        if modulus <= u8_max {
            (quote! { u8  }, "u8")
        } else if modulus <= u16_max {
            (quote! { u16 }, "u16")
        } else if modulus <= u32_max {
            (quote! { u32 }, "u32")
        } else if modulus <= u64_max {
            (quote! { u64 }, "u64")
        } else {
            (quote! { u128 }, "u128")
        }
    };

    let name = &ast.ident;
    let gen = quote! {
        impl SmallFpConfig for #name {
            type T = #ty;

            const MODULUS: Self::T = #modulus as Self::T;

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


            fn inverse(a: &SmallFp<Self>) -> Option<SmallFp<Self>> {
                None
            }

            fn from_bigint(_: BigInt<2>) -> Option<SmallFp<Self>> {
                None
            }

            fn into_bigint(_: SmallFp<Self>) -> BigInt<2> {
                BigInt::new([0, 0])
            }
        }
    };
    gen.into()
}
