use ark_ff::{BigInt, SqrtPrecomputation};
use fields_macro::SmallFpConfig;
use space_efficient_sumcheck::fields::small_fp_backend::SmallFp;
use space_efficient_sumcheck::fields::small_fp_backend::SmallFpConfig;

#[derive(SmallFpConfig)]
#[modulus = "1234567890"]
#[generator = "2"]
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
