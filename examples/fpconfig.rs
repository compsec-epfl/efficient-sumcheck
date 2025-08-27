use ark_ff::{BigInt, SqrtPrecomputation};
use fields_macro::SmallFpConfig;
use space_efficient_sumcheck::fields::small_fp_backend::SmallFp;
use space_efficient_sumcheck::fields::small_fp_backend::SmallFpConfig;

#[derive(SmallFpConfig)]
#[modulus = "2147483647"]
#[generator = "7"]
#[backend = "standard"]
struct SmallField;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_assign_test() {
        let mut a = SmallField::new(20);
        let b = SmallField::new(10);
        let c = SmallField::new(30);
        a += b;
        assert_eq!(a.value, c.value);

        let mut a = SmallField::new(SmallField::MODULUS - 1);
        let b = SmallField::new(2);
        a += b;
        assert_eq!(a.value, 1);

        // adding zero
        let mut a = SmallField::new(42);
        let b = SmallField::ZERO;
        a += b;
        assert_eq!(a.value, 42);

        // max values
        let mut a = SmallField::new(SmallField::MODULUS - 1);
        let b = SmallField::new(SmallField::MODULUS - 1);
        a += b;
        assert_eq!(a.value, SmallField::MODULUS - 2);

        // adding one to maximum
        let mut a = SmallField::new(SmallField::MODULUS - 1);
        let b = SmallField::ONE;
        a += b;
        assert_eq!(a.value, 0);
    }

    #[test]
    fn sub_assign_test() {
        let mut a = SmallField::new(30);
        let b = SmallField::new(10);
        let c = SmallField::new(20);
        a -= b;
        assert_eq!(a.value, c.value);

        let mut a = SmallField::new(5);
        let b = SmallField::new(10);
        a -= b;
        assert_eq!(a.value, SmallField::MODULUS - 5);

        // subtracting zero
        let mut a = SmallField::new(42);
        let b = SmallField::ZERO;
        a -= b;
        assert_eq!(a.value, 42);

        // subtracting from zero
        let mut a = SmallField::ZERO;
        let b = SmallField::new(1);
        a -= b;
        assert_eq!(a.value, SmallField::MODULUS - 1);

        // self subtraction
        let mut a = SmallField::new(42);
        let b = SmallField::new(42);
        a -= b;
        assert_eq!(a.value, 0);

        // maximum minus one
        let mut a = SmallField::new(SmallField::MODULUS - 1);
        let b = SmallField::ONE;
        a -= b;
        assert_eq!(a.value, SmallField::MODULUS - 2);
    }

    #[test]
    fn mul_assign_test() {
        let mut a = SmallField::new(5);
        let b = SmallField::new(10);
        let c = SmallField::new(50);
        a *= b;
        assert_eq!(a.value, c.value);

        let mut a = SmallField::new(SmallField::MODULUS / 2);
        let b = SmallField::new(3);
        a *= b;
        assert_eq!(a.value, (SmallField::MODULUS / 2) * 3 % SmallField::MODULUS);

        // multiply by zero
        let mut a = SmallField::new(42);
        let b = SmallField::ZERO;
        a *= b;
        assert_eq!(a.value, 0);

        // multiply by one
        let mut a = SmallField::new(42);
        let b = SmallField::ONE;
        a *= b;
        assert_eq!(a.value, 42);

        // maximum values
        let mut a = SmallField::new(SmallField::MODULUS - 1);
        let b = SmallField::new(SmallField::MODULUS - 1);
        a *= b;
        assert_eq!(a.value, 1); // (p-1)*(p-1) = p^2 - 2p + 1 ≡ 1 (mod p)
    }

    #[test]
    fn neg_in_place_test() {
        let mut a = SmallField::new(10);
        SmallField::neg_in_place(&mut a);
        assert_eq!(a.value, SmallField::MODULUS - 10);

        let mut a = SmallField::ZERO;
        SmallField::neg_in_place(&mut a);
        assert_eq!(a.value, 0);

        // negate maximum
        let mut a = SmallField::new(SmallField::MODULUS - 1);
        SmallField::neg_in_place(&mut a);
        assert_eq!(a.value, 1);

        // Edge double negation
        let mut a = SmallField::new(42);
        let original = a.value;
        SmallField::neg_in_place(&mut a);
        SmallField::neg_in_place(&mut a);
        assert_eq!(a.value, original);

        // negate one
        let mut a = SmallField::ONE;
        SmallField::neg_in_place(&mut a);
        assert_eq!(a.value, SmallField::MODULUS - 1);
    }

    #[test]
    fn double_in_place_test() {
        let mut a = SmallField::new(10);
        SmallField::double_in_place(&mut a);
        assert_eq!(a.value, 20);

        let mut a = SmallField::new(SmallField::MODULUS - 1);
        SmallField::double_in_place(&mut a);
        assert_eq!(a.value, SmallField::MODULUS - 2);

        // double zero
        let mut a = SmallField::ZERO;
        SmallField::double_in_place(&mut a);
        assert_eq!(a.value, 0);

        // double maximum/2 + 1 (should wrap)
        if SmallField::MODULUS > 2 {
            let mut a = SmallField::new(SmallField::MODULUS / 2 + 1);
            SmallField::double_in_place(&mut a);
            assert_eq!(
                a.value,
                (SmallField::MODULUS / 2 + 1) * 2 % SmallField::MODULUS
            );
        }

        // double one
        let mut a = SmallField::ONE;
        SmallField::double_in_place(&mut a);
        assert_eq!(a.value, 2);
    }

    #[test]
    fn square_in_place_test() {
        let mut a = SmallField::new(5);
        let b = SmallField::new(25);
        SmallField::square_in_place(&mut a);
        assert_eq!(a.value, b.value);

        let mut a = SmallField::new(SmallField::MODULUS - 1);
        SmallField::square_in_place(&mut a);
        assert_eq!(a.value, 1);

        // square zero
        let mut a = SmallField::ZERO;
        SmallField::square_in_place(&mut a);
        assert_eq!(a.value, 0);

        // square one
        let mut a = SmallField::ONE;
        SmallField::square_in_place(&mut a);
        assert_eq!(a.value, 1);
    }

    #[test]
    fn zero_inverse() {
        let zero = SmallField::ZERO;
        assert!(SmallField::inverse(&zero).is_none())
    }

    #[test]
    fn test_specific_inverse() {
        let mut val = SmallField::new(17);
        let val_inv = SmallField::inverse(&val);
        SmallField::mul_assign(&mut val, &val_inv.unwrap());
        assert_eq!(val, SmallField::ONE);
    }

    #[test]
    fn test_inverse() {
        // inverse of 1
        let one = SmallField::ONE;
        let one_inv = SmallField::inverse(&one).unwrap();
        assert_eq!(one_inv, SmallField::ONE);

        // inverse of p-1 (which should be p-1 since (p-1)^2 ≡ 1 mod p)
        let neg_one = SmallField::new(SmallField::MODULUS - 1);
        let neg_one_inv = SmallField::inverse(&neg_one).unwrap();
        assert_eq!(neg_one_inv.value, SmallField::MODULUS - 1);

        for i in 1..100 {
            let val = SmallField::new(i);
            if let Some(inv) = SmallField::inverse(&val) {
                let mut product = val;
                SmallField::mul_assign(&mut product, &inv);
                assert_eq!(product, SmallField::ONE, "Failed for value {}", i);
            }
        }

        // inverse property: inv(inv(x)) = x
        let test_val = SmallField::new(42 % SmallField::MODULUS);
        if test_val.value != 0 {
            let inv1 = SmallField::inverse(&test_val).unwrap();
            let inv2 = SmallField::inverse(&inv1).unwrap();
            assert_eq!(test_val, inv2);
        }

        // inverse is multiplicative: inv(ab) = inv(a) * inv(b)
        let a = SmallField::new(7 % SmallField::MODULUS);
        let b = SmallField::new(11 % SmallField::MODULUS);
        if a.value != 0 && b.value != 0 {
            let mut ab = a;
            SmallField::mul_assign(&mut ab, &b);

            let inv_ab = SmallField::inverse(&ab).unwrap();
            let inv_a = SmallField::inverse(&a).unwrap();
            let inv_b = SmallField::inverse(&b).unwrap();

            let mut inv_a_times_inv_b = inv_a;
            SmallField::mul_assign(&mut inv_a_times_inv_b, &inv_b);

            assert_eq!(inv_ab, inv_a_times_inv_b);
        }
    }

    #[test]
    fn test_field_axioms() {
        // Test additive identity
        let a = SmallField::new(42 % SmallField::MODULUS);
        let b = SmallField::new(73 % SmallField::MODULUS);
        // commutativity of multiplication
        let mut a_times_b = a;
        let mut b_times_a = b;
        SmallField::mul_assign(&mut a_times_b, &b);
        SmallField::mul_assign(&mut b_times_a, &a);
        assert_eq!(a_times_b, b_times_a);

        // associativity of addition: (a + b) + c = a + (b + c)
        let c = SmallField::new(91 % SmallField::MODULUS);
        let mut ab_plus_c = a;
        SmallField::add_assign(&mut ab_plus_c, &b);
        SmallField::add_assign(&mut ab_plus_c, &c);

        let mut a_plus_bc = a;
        let mut bc = b;
        SmallField::add_assign(&mut bc, &c);
        SmallField::add_assign(&mut a_plus_bc, &bc);

        assert_eq!(ab_plus_c, a_plus_bc);

        // distributivity: a * (b + c) = a * b + a * c
        let mut a_times_bc = a;
        let mut bc = b;
        SmallField::add_assign(&mut bc, &c);
        SmallField::mul_assign(&mut a_times_bc, &bc);

        let mut ab_plus_ac = a;
        SmallField::mul_assign(&mut ab_plus_ac, &b);
        let mut ac = a;
        SmallField::mul_assign(&mut ac, &c);
        SmallField::add_assign(&mut ab_plus_ac, &ac);

        assert_eq!(a_times_bc, ab_plus_ac);
    }

    #[test]
    fn test_sum_of_products() {
        let a = [SmallField::new(2), SmallField::new(3), SmallField::new(5)];
        let b = [SmallField::new(7), SmallField::new(11), SmallField::new(13)];
        let result = SmallField::sum_of_products(&a, &b);
        assert_eq!(result.value, 112 % SmallField::MODULUS);

        let a = [SmallField::ZERO, SmallField::new(3), SmallField::ZERO];
        let b = [SmallField::new(7), SmallField::new(11), SmallField::new(13)];
        let result = SmallField::sum_of_products(&a, &b);
        assert_eq!(result.value, 33 % SmallField::MODULUS);
    }
}
