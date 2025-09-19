use super::*;
use crate::utils::{compute_two_adic_root_of_unity, compute_two_adicity};

pub fn backend_impl(
    ty: proc_macro2::TokenStream,
    modulus: u128,
    generator: u128,
    _suffix: &str,
) -> proc_macro2::TokenStream {
    let k_bits = 128 - modulus.leading_zeros();
    let r: u128 = 1u128 << k_bits;
    let r_mod_n = r % modulus;
    let r_mask = r - 1;

    let n_prime = mod_inverse_pow2(modulus, k_bits);
    let one_mont = r_mod_n;
    let generator_mont = (generator % modulus) * (r_mod_n % modulus) % modulus;

    let two_adicity = compute_two_adicity(modulus);
    let two_adic_root = compute_two_adic_root_of_unity(modulus, generator, two_adicity);
    let two_adic_root_mont = (two_adic_root * r_mod_n) % modulus;

    let (from_bigint_impl, into_bigint_impl) =
        generate_montgomery_bigint_casts(modulus, k_bits, r_mod_n);

    quote! {
        type T = #ty;
        const MODULUS: Self::T = #modulus as Self::T;
        const MODULUS_128: u128 = #modulus;
        const GENERATOR: SmallFp<Self> = SmallFp::new(#generator_mont as Self::T);
        const ZERO: SmallFp<Self> = SmallFp::new(0 as Self::T);
        const ONE: SmallFp<Self> = SmallFp::new(#one_mont as Self::T);


        const TWO_ADICITY: u32 = #two_adicity;
        const TWO_ADIC_ROOT_OF_UNITY: SmallFp<Self> = SmallFp::new(#two_adic_root_mont as Self::T);

        const SQRT_PRECOMP: Option<SqrtPrecomputation<SmallFp<Self>>> = None;

        fn add_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            a.value = match a.value.overflowing_add(b.value) {
                (val, false) => val % Self::MODULUS,
                (val, true) => (Self::T::MAX - Self::MODULUS + 1 + val) % Self::MODULUS,
            };
        }

        fn sub_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            if a.value >= b.value {
                a.value -= b.value;
            } else {
                a.value = Self::MODULUS - (b.value - a.value);
            }
        }

        fn double_in_place(a: &mut SmallFp<Self>) {
            let tmp = *a;
            Self::add_assign(a, &tmp);
        }

        fn neg_in_place(a: &mut SmallFp<Self>) {
            if a.value == (0 as Self::T) {
                a.value = 0 as Self::T;
            } else {
                a.value = Self::MODULUS - a.value;
            }
        }

        fn safe_mul(a: Self::T, b: Self::T) -> Self::T {
            let a_u128 = a as u128;
            let b_u128 = b as u128;

            // t = a * b
            let t = a_u128.wrapping_mul(b_u128);

            // m = (t * n')
            let m = t.wrapping_mul(#n_prime) & #r_mask;

            // u = (t + m * n) * r^{-1}
            let mn = (m as u128).wrapping_mul(#modulus);
            let t_plus_mn = t.wrapping_add(mn);
            let mut u = t_plus_mn >> #k_bits;

            if u >= #modulus {
                u -= #modulus;
            }
            u as Self::T
        }

        fn mul_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            a.value = Self::safe_mul(a.value, b.value);
        }

        fn sum_of_products<const T: usize>(
            a: &[SmallFp<Self>; T],
            b: &[SmallFp<Self>; T],) -> SmallFp<Self> {
            let mut acc = SmallFp::new(0 as Self::T);
            for (x, y) in a.iter().zip(b.iter()) {
                let prod = SmallFp::new(Self::safe_mul(x.value, y.value));
                Self::add_assign(&mut acc, &prod);
            }
            acc
        }

        fn square_in_place(a: &mut SmallFp<Self>) {
            a.value = Self::safe_mul(a.value, a.value);
        }

        fn inverse(a: &SmallFp<Self>) -> Option<SmallFp<Self>> {
            if a.value == 0 {
                return None;
            }

            let exponent = Self::MODULUS - 2;
            let mut result = Self::ONE;
            let mut base = *a;
            let mut exp = exponent;

            while exp > 0 {
                if exp & 1 == 1 {
                    Self::mul_assign(&mut result, &base);
                }

                Self::square_in_place(&mut base);
                exp >>= 1;
            }

            Some(result)
        }

        #from_bigint_impl

        #into_bigint_impl
    }
}

fn mod_inverse_pow2(n: u128, bits: u32) -> u128 {
    let mut inv = 1u128;
    for _ in 0..bits {
        inv = inv.wrapping_mul(2u128.wrapping_sub(n.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

fn generate_montgomery_bigint_casts(
    modulus: u128,
    _k_bits: u32,
    r_mod_n: u128,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let r2 = (r_mod_n * r_mod_n) % modulus;
    (
        quote! {
            //* Convert from standard representation to Montgomery space
            fn from_bigint(a: BigInt<2>) -> Option<SmallFp<Self>> {
                let val = (a.0[0] as u128) + ((a.0[1] as u128) << 64);
                let reduced_val = val % #modulus;
                // Convert to Montgomery space by multiplying by R²
                let mont_value = Self::safe_mul(reduced_val as Self::T, #r2 as Self::T);
                Some(SmallFp::new(mont_value))
            }
        },
        quote! {
            //* Convert from Montgomery space to standard representation
            fn into_bigint(a: SmallFp<Self>) -> BigInt<2> {
                // Exit Montgomery space by multiplying by 1
                let std_value = Self::safe_mul(a.value, 1 as Self::T);
                ark_ff::BigInt([std_value as u64, 0])
            }
        },
    )
}

pub fn new(modulus: u128, _ty: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let k_bits = 128 - modulus.leading_zeros();
    let r: u128 = 1u128 << k_bits;
    let r2 = (r * r) % modulus;

    quote! {
        pub fn new(value: <Self as SmallFpConfig>::T) -> SmallFp<Self> {
            let reduced_value = value % <Self as SmallFpConfig>::MODULUS;
            let mont_value = <Self as SmallFpConfig>::safe_mul(reduced_value, #r2 as <Self as SmallFpConfig>::T);
            SmallFp::new(mont_value)
        }

        pub fn exit(a: &mut SmallFp<Self>) {
            let std_val = <Self as SmallFpConfig>::safe_mul(a.value, 1 as <Self as SmallFpConfig>::T);
            a.value = std_val;
        }

    }
}
