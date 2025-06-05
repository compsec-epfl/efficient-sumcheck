use core::convert::Into;
use proc_macro::TokenStream;
use quote::quote;

use proc_macro2::Span;
use syn::{Expr, ExprLit, Lit, LitInt, Meta};

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

// #[proc_macro_derive(FpConfig2, attributes(modulus, generator))]
// pub fn fp_config(input: TokenStream) -> TokenStream {
//     let ast: syn::DeriveInput = syn::parse(input).unwrap();

//     // Fetch the modulus and generator attributes
//     let modulus: u128 = fetch_attr("modulus", &ast.attrs)
//         .expect("Please supply a modulus attribute")
//         .parse()
//         .expect("Modulus should be a number");

//     let generator: u128 = fetch_attr("generator", &ast.attrs)
//         .expect("Please supply a generator attribute")
//         .parse()
//         .expect("Generator should be a number");

//     // Determine the smallest type for the modulus
//     let ty = {
//         let u8_max = u128::from(u8::MAX);
//         let u16_max = u128::from(u16::MAX);
//         let u32_max = u128::from(u32::MAX);
//         let u64_max = u128::from(u64::MAX);

//         if modulus <= u8_max {
//             quote! { u8 }
//         } else if modulus <= u16_max {
//             quote! { u16 }
//         } else if modulus <= u32_max {
//             quote! { u32 }
//         } else if modulus <= u64_max {
//             quote! { u64 }
//         } else {
//             quote! { u128 }
//         }
//     };

//     let modulus_str = modulus.to_string();
//     let generator_str = generator.to_string();

//     // Generate the implementation of the trait
//     let expanded = quote! {
//         impl FpConfig2 for #ast.ident {
//             const MODULUS: #ty = #modulus_str as #ty;
//             const GENERATOR: #ty = #generator_str as #ty;

//             fn hello_macro() {
//                 println!("Hello, Macro! My name is {}!", stringify!(#ast.ident));
//             }
//         }
//     };

//     TokenStream::from(expanded)
// }

#[proc_macro_derive(FpConfig2, attributes(modulus, generator))]
pub fn fp_config(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    // Fetch the modulus and generator attributes
    let modulus: u128 = fetch_attr("modulus", &ast.attrs)
        .expect("Please supply a modulus attribute")
        .parse()
        .expect("Modulus should be a number");

    let generator: u128 = fetch_attr("generator", &ast.attrs)
        .expect("Please supply a generator attribute")
        .parse()
        .expect("Generator should be a number");

    // Determine the smallest type for the modulus
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

    // Build the trait implementation
    let name = &ast.ident;
    let gen = quote! {

        impl FpConfig2 for #name {
            type UInt = #ty;

            const MODULUS: Self::UInt = #modulus as Self::UInt;
            const GENERATOR: Self::UInt = #generator as Self::UInt;

            fn hello_macro() {
                println!("Hello, Macro! My name is {}!", stringify!(#name));
            }

            fn add_assign(a: &mut Self::UInt, b: &Self::UInt) {
                *a = a.wrapping_add(*b);
            }

        }
    };
    gen.into()
}
