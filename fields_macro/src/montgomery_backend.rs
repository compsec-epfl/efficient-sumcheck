use super::*;

pub fn montgomery_backend_impl(
    ty: proc_macro2::TokenStream,
    modulus: u128,
    generator: u128,
    suffix: &str,
) -> proc_macro2::TokenStream {
    // TODO: think about the choice of R
    // let r = if suffix == "u128" {
    //     1u128 << 127 % modulus
    // } else {
    //     1u128 << 64 % modulus
    // };

    let mut r = 1u128;
    while r << 1 < u128::MAX && r <= modulus {
        r <<= 1;
    }

    r = r % modulus;
    let r2 = (r * r) % modulus;
    let mod_inv = compute_n_prime(modulus, 64);
    // let mod_inv = compute_n_prime(modulus, if suffix == "u128" { 128 } else { 64 });

    let one_mont = r;
    let generator_mont = (generator * r) % modulus;

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
            let mut base = a.value;
            let mut exp = Self::MODULUS - 2;
            let mut acc = Self::ONE.value;
            while exp > 0 {
                if (exp & 1) == 1 {
                    acc = Self::safe_mul(acc, base);
                }
                base = Self::safe_mul(base, base);
                exp >>= 1;
            }
            Some(SmallFp::new(acc))
        }

        #from_bigint_impl

        #into_bigint_impl
    }
}

const fn compute_n_prime(n: u128, r_bits: u32) -> u128 {
    let r = if r_bits == 128 {
        u128::MAX
    } else {
        (1u128 << r_bits) - 1
    };

    let mut a = n;
    let mut b = r + 1; // 2^r_bits
    let mut x0 = 1u128;
    let mut x1 = 0u128;

    while a > 1 {
        let q = a / b;
        let t = b;
        b = a % b;
        a = t;
        let t = x1;
        x1 = x0.wrapping_sub(q.wrapping_mul(x1));
        x0 = t;
    }

    x0.wrapping_neg()
        & if r_bits == 128 {
            u128::MAX
        } else {
            (1u128 << r_bits) - 1
        }
}
