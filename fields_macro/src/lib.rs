use core::convert::Into;
use montgomery_backend::montgomery_backend_impl;
use proc_macro::TokenStream;
use quote::quote;
use standard_backend::standard_backend_impl;

use syn::{Expr, ExprLit, Lit, Meta};

extern crate proc_macro;
mod montgomery_backend;
mod standard_backend;
mod utils;

fn fetch_attr(name: &str, attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        match attr.meta {
            // If the attribute's path matches `name`, and if the attribute is of
            // the form `#[name = "value"]`, return `value`
            Meta::NameValue(ref nv) if nv.path.is_ident(name) => {
                // Extract and return the string value.
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

#[proc_macro_derive(SmallFpConfig, attributes(modulus, generator, backend))]
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

    let backend: String = fetch_attr("backend", &ast.attrs)
        .expect("Please supply a backend attribute")
        .parse()
        .expect("Backend should be a string");

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

    let backend_impl = match backend.as_str() {
        "standard" => standard_backend_impl(ty.clone(), modulus, generator, suffix),
        "montgomery" => montgomery_backend_impl(ty.clone(), modulus, generator, suffix),
        _ => panic!("Unknown backend type"),
    };

    let name = &ast.ident;
    let gen = quote! {
        impl SmallFpConfig for #name {
            #backend_impl
        }

        impl #name {
            pub fn new(value: <Self as SmallFpConfig>::T) -> SmallFp<Self> {
                SmallFp::new(value % <Self as SmallFpConfig>::MODULUS)
            }
        }
    };

    gen.into()
}
