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
