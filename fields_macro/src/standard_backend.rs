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
                    let val = (a.0[0] as u128) + ((a.0[1] as u128) << 64);
                    if val >= Self::MODULUS_128 {
                        None
                    } else {
                        Some(SmallFp::new(val as Self::T))
                    }
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
                    if a.0[1] != 0 || a.0[0] >= (Self::MODULUS as u64) {
                        None
                    } else {
                        Some(SmallFp::new(a.0[0] as Self::T))
                    }
                }
            },
            quote! {
                fn into_bigint(a: SmallFp<Self>) -> BigInt<2> {
                    ark_ff::BigInt([a.value as u64, 0])
                }
            },
        )
    };

    quote! {
        type T = #ty;
        const MODULUS: Self::T = #modulus as Self::T;
        const MODULUS_128: u128 = #modulus;
        const GENERATOR: SmallFp<Self> = SmallFp::new(#generator as Self::T);
        const ZERO: SmallFp<Self> = SmallFp::new(0 as Self::T);
        const ONE: SmallFp<Self> = SmallFp::new(1 as Self::T);

        // TODO: complete this
        // copy approach from here: https://github.com/arkworks-rs/algebra/blob/851d4680491ed97fb09b5410893c2e12377b2bec/ff/src/biginteger/mod.rs#L215
        const TWO_ADICITY: u32 = 1;
        const TWO_ADIC_ROOT_OF_UNITY: SmallFp<Self> = SmallFp::new(1 as Self::T);

        // Todo: precompute square roots
        const SQRT_PRECOMP: Option<SqrtPrecomputation<SmallFp<Self>>> = None;

        fn add_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>) {
            let sum = a.value.overflowing_add(b.value);
            a.value = match sum {
                (val, false) => val % Self::MODULUS,
                (val, true) => (val +Self::T::MAX - Self::MODULUS + val) % Self::MODULUS,
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
            //* Note: This might be faster using bitshifts
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

        /*
        To avoid overflow, split into halves:
        a = a1*C + a0, b = b1*C + b0
        a*b = a1*b1*C^2 + (a1*b0 + a0*b1)*C + a0*b0

        Each term is computed modulo N to prevent overflow.
        */
        fn safe_mul(a: Self::T, b: Self::T) -> Self::T {
            match (a as u128).overflowing_mul(b as u128) {
                (val, false) => (val % Self::MODULUS_128) as Self::T,
                (val, true) => {
                    let C = (1u128 << 64 - 1) % Self::MODULUS_128;
                    let C2 = (C * C) % Self::MODULUS_128;

                    let a1 = (a as u128) >> 64;
                    let a0 = (a as u128) & ((1u128 << 64) - 1);
                    let b1 = (b as u128) >> 64;
                    let b0 = (b as u128) & ((1u128 << 64) - 1);

                    let a1b1 = (a1 * b1) % Self::MODULUS_128;
                    let a1b0 = (a1 * b0) % Self::MODULUS_128;
                    let a0b1 = (a0 * b1) % Self::MODULUS_128;
                    let a0b0 = (a0 * b0) % Self::MODULUS_128;

                    let mut acc = 0u128;
                    acc = (acc + ((a1b1 as u128) * (C2 as u128)) % Self::MODULUS_128) % Self::MODULUS_128;
                    let cross_sum = (a1b0 + a0b1) % Self::MODULUS_128;
                    acc = (acc + ((cross_sum as u128) * (C as u128)) % Self::MODULUS_128) % Self::MODULUS_128;
                    acc = (acc + a0b0) % Self::MODULUS_128;

                    (acc % Self::MODULUS_128) as Self::T
                }
            }
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
            let tmp = *a;
            Self::mul_assign(a, &tmp);
        }

        fn inverse(a: &SmallFp<Self>) -> Option<SmallFp<Self>> {
            if a.value == 0 {
                return None;
            }

            let mut base = a.value;
            let mut exp = Self::MODULUS - 2;
            let mut acc = 1 as Self::T;

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
