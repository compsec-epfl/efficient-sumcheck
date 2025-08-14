use super::*;

pub fn montgomery_backend_impl(
    _ty: proc_macro2::TokenStream,
    _modulus: u128,
    _generator: u128,
) -> proc_macro2::TokenStream {
    quote! {
        compile_error!("Montgomery backend is not yet implemented");
    }
}
