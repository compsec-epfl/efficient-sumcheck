use std::str::FromStr;

use num_bigint::{BigInt, Sign};
use num_traits::Num;
use proc_macro::TokenStream;
use syn::{Expr, Lit};

pub(crate) fn parse_string(input: TokenStream) -> Option<String> {
    let input: Expr = syn::parse(input).unwrap();
    let input = if let Expr::Group(syn::ExprGroup { expr, .. }) = input {
        expr
    } else {
        panic!("could not parse");
    };
    match *input {
        Expr::Lit(expr_lit) => match expr_lit.lit {
            Lit::Str(s) => Some(s.value()),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn str_to_limbs(num: &str) -> (bool, Vec<String>) {
    let (sign, limbs) = str_to_limbs_u64(num);
    (sign, limbs.into_iter().map(|l| format!("{l}u64")).collect())
}

pub(crate) fn str_to_limbs_u64(num: &str) -> (bool, Vec<u64>) {
    let is_negative = num.starts_with('-');
    let num = if is_negative { &num[1..] } else { num };
    let number = if num.starts_with("0x") || num.starts_with("0X") {
        // We are in hexadecimal
        BigInt::from_str_radix(&num[2..], 16)
    } else if num.starts_with("0o") || num.starts_with("0O") {
        // We are in octal
        BigInt::from_str_radix(&num[2..], 8)
    } else if num.starts_with("0b") || num.starts_with("0B") {
        // We are in binary
        BigInt::from_str_radix(&num[2..], 2)
    } else {
        // We are in decimal
        BigInt::from_str(num)
    }
    .expect("could not parse to bigint");
    let number = if is_negative { -number } else { number };
    let (sign, digits) = number.to_radix_le(16);

    let limbs = digits
        .chunks(16)
        .map(|chunk| {
            let mut this = 0u64;
            for (i, hexit) in chunk.iter().enumerate() {
                this += (*hexit as u64) << (4 * i);
            }
            this
        })
        .collect::<Vec<_>>();

    let sign_is_positive = sign != Sign::Minus;
    (sign_is_positive, limbs)
}

// Compute the largest integer `s` such that `N - 1 = 2**s * t` for odd `t`.
pub const fn compute_two_adicity(modulus: u128) -> u32 {
    assert!(modulus % 2 == 1, "Modulus must be odd");
    assert!(modulus > 1, "Modulus must be greater than 1");

    let mut n_minus_1 = modulus - 1;
    let mut two_adicity = 0;

    while n_minus_1 % 2 == 0 {
        n_minus_1 /= 2;
        two_adicity += 1;
    }
    two_adicity
}

const fn mod_add(x: u128, y: u128, modulus: u128) -> u128 {
    if x >= modulus - y {
        x - (modulus - y)
    } else {
        x + y
    }
}

const fn safe_mul_const(a: u128, b: u128, modulus: u128) -> u128 {
    match a.overflowing_mul(b) {
        (val, false) => val % modulus,
        (_, true) => {
            let mut result = 0u128;
            let mut base = a % modulus;
            let mut exp = b;

            while exp > 0 {
                if exp & 1 == 1 {
                    result = mod_add(result, base, modulus);
                }
                base = mod_add(base, base, modulus);
                exp >>= 1;
            }
            result
        }
    }
}

// Two adicity root of unity `w` is defined as `w = g^((N-1)/2^s)` where `s` is two adidcity
// Therefore `w^(2^s) = 1 mod N`
pub const fn compute_two_adic_root_of_unity(
    modulus: u128,
    generator: u128,
    two_adicity: u32,
) -> u128 {
    let mut exp = (modulus - 1) >> two_adicity;
    let mut base = generator % modulus;
    let mut result = 1u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = safe_mul_const(result, base, modulus);
        }
        base = safe_mul_const(base, base, modulus);
        exp /= 2;
    }
    result
}
