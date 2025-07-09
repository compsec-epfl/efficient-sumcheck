use ark_ff::{BigInt, SqrtPrecomputation};
use fields_macro::SmallFpConfig;
use space_efficient_sumcheck::fields::small_fp_backend::SmallFp;
use space_efficient_sumcheck::fields::small_fp_backend::SmallFpConfig;

#[derive(SmallFpConfig)]
#[modulus = "2147483647"]
#[generator = "7"]
struct SomeField;

fn main() {
    println!(
        "MOD: {} GENERATOR: {}",
        SomeField::MODULUS,
        SomeField::GENERATOR
    );
    let mut a = SomeField::ONE;
    let b = SomeField::ONE;
    let c = SomeField::new(4);
    a += b;
    a += c;
    println!("{}", a);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{One, Zero};

    #[test]
    fn add_assign_test() {
        let mut a = SomeField::new(10);
        let b = SomeField::new(20);
        a += b;
        assert_eq!(a.value, 30);

        let mut a = SomeField::new(SomeField::MODULUS - 1);
        let b = SomeField::new(2);
        a += b;
        assert_eq!(a.value, 1);
    }

    #[test]
    fn sub_assign_test() {
        let mut a = SomeField::new(30);
        let b = SomeField::new(10);
        a -= b;
        assert_eq!(a.value, 20);

        let mut a = SomeField::new(5);
        let b = SomeField::new(10);
        a -= b;
        assert_eq!(a.value, SomeField::MODULUS - 5);
    }

    #[test]
    fn mul_assign_test() {
        let mut a = SomeField::new(5);
        let b = SomeField::new(10);
        a *= b;
        assert_eq!(a.value, 50);

        let mut a = SomeField::new(SomeField::MODULUS / 2);
        let b = SomeField::new(3);
        a *= b;
        assert_eq!(a.value, (SomeField::MODULUS / 2) * 3 % SomeField::MODULUS);
    }

    #[test]
    fn neg_in_place_test() {
        let mut a = SomeField::new(10);
        SomeField::neg_in_place(&mut a);
        assert_eq!(a.value, SomeField::MODULUS - 10);

        let mut a = SomeField::ZERO;
        SomeField::neg_in_place(&mut a);
        assert_eq!(a.value, 0);
    }

    #[test]
    fn double_in_place_test() {
        let mut a = SomeField::new(10);
        SomeField::double_in_place(&mut a);
        assert_eq!(a.value, 20);

        let mut a = SomeField::new(SomeField::MODULUS - 1);
        SomeField::double_in_place(&mut a);
        assert_eq!(a.value, SomeField::MODULUS - 2);
    }

    #[test]
    fn square_in_place_test() {
        let mut a = SomeField::new(5);
        SomeField::square_in_place(&mut a);
        assert_eq!(a.value, 25);

        let mut a = SomeField::new(SomeField::MODULUS - 1);
        SomeField::square_in_place(&mut a);
        assert_eq!(a.value, 1);
    }

    #[test]
    fn zero_inverse() {
        let zero = SomeField::ZERO;
        assert!(SomeField::inverse(&zero).is_none())
    }

    #[test]
    fn test_specific_inverse() {
        let mut val = SomeField::new(17);
        let val_inv = SomeField::inverse(&val);
        SomeField::mul_assign(&mut val, &val_inv.unwrap());
        assert_eq!(val, SomeField::ONE);
    }
}
