use super::*;

pub fn backend_impl(
    ty: proc_macro2::TokenStream,
    modulus: u128,
    generator: u128,
    suffix: &str,
) -> proc_macro2::TokenStream {
    // TODO: Panic if the next power cannot fit u128,
    let r = modulus.next_power_of_two();
    let r_mod_n = r % modulus;
    let r2 = (r_mod_n * r_mod_n) % modulus;
    // TODO: fix this
    let mod_inv = 128;
    let one_mont = r_mod_n;
    let generator_mont = (generator * r_mod_n) % modulus;

    let (from_bigint_impl, into_bigint_impl) = if suffix == "u128" {
        (
            quote! {
                fn from_bigint(a: BigInt<2>) -> Option<SmallFp<Self>> {
                    let val = (a.0[0] as u128) + ((a.0[1] as u128) << 64);
                    if val >= Self::MODULUS_128 {
                        None
                    } else {
                        let mont_val = Self::safe_mul(val as Self::T, #r2 as Self::T);
                        Some(SmallFp::new(mont_val))
                    }
                }
            },
            quote! {
                fn into_bigint(a: SmallFp<Self>) -> BigInt<2> {
                    let val = Self::safe_mul(a.value, 1 as Self::T) as u128;
                    ark_ff::BigInt([(val as u64), (val >> 64) as u64])
                }
            },
        )
    } else {
        (
            quote! {
                fn from_bigint(a: BigInt<2>) -> Option<SmallFp<Self>> {
                    if a.0[1] != 0 || a.0[0] >= (Self::MODULUS as u64) {
                        None
                    } else {
                        let mont_val = Self::safe_mul(a.0[0] as Self::T, #r2 as Self::T);
                        Some(SmallFp::new(mont_val))
                    }
                }
            },
            quote! {
                fn into_bigint(a: SmallFp<Self>) -> BigInt<2> {
                    let val = Self::safe_mul(a.value, 1 as Self::T) as u64;
                    ark_ff::BigInt([val, 0])
                }
            },
        )
    };

    quote! {
        type T = #ty;
        const MODULUS: Self::T = #modulus as Self::T;
        const MODULUS_128: u128 = #modulus;

        const GENERATOR: SmallFp<Self> = SmallFp::new(#generator_mont as Self::T);
        const ZERO: SmallFp<Self> = SmallFp::new(0 as Self::T);
        const ONE: SmallFp<Self> = SmallFp::new(#one_mont as Self::T);

        // TODO: complete this - need Montgomery form
        const TWO_ADICITY: u32 = 1;
        const TWO_ADIC_ROOT_OF_UNITY: SmallFp<Self> = SmallFp::new(1 as Self::T);

        // Todo: precompute square roots - need Montgomery form
        const SQRT_PRECOMP: Option<SqrtPrecomputation<SmallFp<Self>>> = None;

        fn add_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            let sum = a.value.overflowing_add(b.value);
            a.value = match sum {
                (val, false) => val % Self::MODULUS,
                (val, true) => (val + Self::T::MAX - Self::MODULUS + val) % Self::MODULUS,
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

            // Compute t = a * b
            let mult = a_u128.wrapping_mul(b_u128);

            (mult >> 64) as Self::T
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

            // double and add
            let exponent = Self::MODULUS - 2;
            let mut result = SmallFp::new(1 as Self::T);
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

fn inverse(a: u128, n: u128) -> Option<u128> {
    if a == 0 {
        return None;
    }

    let mut base = a;
    let mut exp = n - 2;
    let mut acc = 1;

    while exp > 0 {
        if (exp & 1) == 1 {
            acc = (acc * base) % n;
        }
        base = (base * base) % n;
        exp >>= 1;
    }
    Some(acc)
}

pub fn new(modulus: u128, ty: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    // TODO: do not recompute this...
    let r: u128 = modulus.next_power_of_two();
    let r_mod_n: u128 = r % modulus;
    let r_inv: u128 = inverse(r_mod_n, modulus).unwrap();

    quote! {
        pub fn new(value: <Self as SmallFpConfig>::T) -> SmallFp<Self> {
                let mont_value: #ty = value * #r_mod_n as #ty;
                SmallFp::new(mont_value)
        }

        pub fn exit(a: SmallFp<Self>) -> SmallFp<Self> {
            let exit: #ty = <Self as SmallFpConfig>::safe_mul(a.value, #r_inv as #ty);
            SmallFp::new(exit % <Self as SmallFpConfig>::MODULUS)
        }
    }
}
