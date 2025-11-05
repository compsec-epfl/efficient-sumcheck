use ark_ff::{BigInt, Fp2, Fp2Config, Fp4, Fp4Config, UniformRand};
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};
use efficient_sumcheck::tests::{
    F128,
    F64 as Goldilocks,
    M31, // SmallF128Mont, SmallF32Mont, SmallF64Mont,
};

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
// pub struct Fp2SmallGoldilocksConfig;

// impl Fp2Config for Fp2SmallGoldilocksConfig {
//     type Fp = SmallF64Mont;

//     // const context: use new_unchecked(BigInt)
//     const NONRESIDUE: SmallF64Mont = SmallF64Mont::new(3);

//     // Arkworks 0.5 expects &'static [Fp]
//     const FROBENIUS_COEFF_FP2_C1: &'static [SmallF64Mont] =
//         &[SmallF64Mont::new(1), SmallF64Mont::new(3)];
// }

// pub type Fp2SmallGoldilocks = Fp2<Fp2SmallGoldilocksConfig>;

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
// pub struct Fp2SmallM31Config;

// impl Fp2Config for Fp2SmallM31Config {
//     type Fp = SmallF32Mont;

//     // Use const_new to build compile-time constants
//     const NONRESIDUE: SmallF32Mont = SmallF32Mont::new(1);

//     // These Frobenius coeffs aren't used for arithmetic benchmarks anyway
//     const FROBENIUS_COEFF_FP2_C1: &'static [SmallF32Mont] = &[
//         SmallF32Mont::new(1),
//         SmallF32Mont::new(3),
//     ];
// }

// pub type Fp2SmallM31 = Fp2<Fp2M31Config>;

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
// pub struct Fp4SmallM31Config;

// impl Fp4Config for Fp4SmallM31Config {
//     type Fp2Config = Fp2SmallM31Config;

//     const NONRESIDUE: Fp2<Fp2SmallM31Config> = Fp2::<Fp2SmallM31Config>::new(
//         SmallF32Mont::new(3),
//         SmallF32Mont::new(0),
//     );

//     // 👇 now a slice of base‐field elements, not Fp2 elements
//     const FROBENIUS_COEFF_FP4_C1: &'static [SmallF32Mont] = &[
//         SmallF32Mont::new(1),
//         SmallF32Mont::new(3),
//         SmallF32Mont::new(9),
//         SmallF32Mont::new(27),
//     ];
// }

// pub type Fp4SmallM31 = Fp4<Fp4SmallM31Config>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp2GoldilocksConfig;

impl Fp2Config for Fp2GoldilocksConfig {
    type Fp = Goldilocks;

    // const context: use new_unchecked(BigInt)
    const NONRESIDUE: Goldilocks = Goldilocks::new_unchecked(BigInt::<1>([3u64]));

    // Arkworks 0.5 expects &'static [Fp]
    const FROBENIUS_COEFF_FP2_C1: &'static [Goldilocks] = &[
        Goldilocks::new_unchecked(BigInt::<1>([1u64])),
        Goldilocks::new_unchecked(BigInt::<1>([3u64])),
    ];
}

pub type Fp2Goldilocks = Fp2<Fp2GoldilocksConfig>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp2M31Config;

impl Fp2Config for Fp2M31Config {
    type Fp = M31;

    // Use const_new to build compile-time constants
    const NONRESIDUE: M31 = M31::new_unchecked(BigInt::<1>([3u64]));

    // These Frobenius coeffs aren't used for arithmetic benchmarks anyway
    const FROBENIUS_COEFF_FP2_C1: &'static [M31] = &[
        M31::new_unchecked(BigInt::<1>([1u64])),
        M31::new_unchecked(BigInt::<1>([3u64])),
    ];
}

pub type Fp2M31 = Fp2<Fp2M31Config>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp4M31Config;

impl Fp4Config for Fp4M31Config {
    type Fp2Config = Fp2M31Config;

    const NONRESIDUE: Fp2<Fp2M31Config> = Fp2::<Fp2M31Config>::new(
        M31::new_unchecked(BigInt::<1>([3u64])),
        M31::new_unchecked(BigInt::<1>([0u64])),
    );

    // 👇 now a slice of base‐field elements, not Fp2 elements
    const FROBENIUS_COEFF_FP4_C1: &'static [M31] = &[
        M31::new_unchecked(BigInt::<1>([1u64])),
        M31::new_unchecked(BigInt::<1>([3u64])),
        M31::new_unchecked(BigInt::<1>([9u64])),
        M31::new_unchecked(BigInt::<1>([27u64])),
    ];
}

pub type Fp4M31 = Fp4<Fp4M31Config>;

pub fn bench_m31_mul(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = M31::rand(&mut rng);
    let b = M31::rand(&mut rng);

    c.bench_function("M31 multiply", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc *= b;
            }
            acc
        });
    });
}

pub fn bench_m31_add(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = M31::rand(&mut rng);
    let b = M31::rand(&mut rng);

    c.bench_function("M31 addition", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc += b;
            }
            acc
        });
    });
}

pub fn bench_fp4_mul(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Fp4M31::rand(&mut rng);
    let b = Fp4M31::rand(&mut rng);

    c.bench_function("Fp4<M31> multiply", |bench| {
        bench.iter(|| {
            let mut acc = a;
            // Chain 1000 multiplications to smooth noise
            for _ in 0..1000 {
                acc *= b;
            }
            acc
        });
    });
}

pub fn bench_fp4_add(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Fp4M31::rand(&mut rng);
    let b = Fp4M31::rand(&mut rng);

    c.bench_function("Fp4<M31> add", |bench| {
        bench.iter(|| {
            let mut acc = a;
            // Chain 1000 multiplications to smooth noise
            for _ in 0..1000 {
                acc += b;
            }
            acc
        });
    });
}

pub fn bench_fp4_base_mult_extension(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Fp4M31::rand(&mut rng);
    let b = M31::rand(&mut rng);

    c.bench_function("Fp4<M31> multiply by basefield", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc.c0.mul_assign_by_basefield(&b); // Fp2 × M31
                acc.c1.mul_assign_by_basefield(&b);
            }
            acc
        });
    });
}

pub fn bench_f128_mul(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = F128::rand(&mut rng);
    let b = F128::rand(&mut rng);

    c.bench_function("F128 multiply", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc *= b;
            }
            acc
        });
    });
}

pub fn bench_f128_add(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = F128::rand(&mut rng);
    let b = F128::rand(&mut rng);

    c.bench_function("F128 addition", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc += b;
            }
            acc
        });
    });
}

pub fn bench_goldilocks_mult(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Goldilocks::rand(&mut rng);
    let b = Goldilocks::rand(&mut rng);

    c.bench_function("Goldilocks mult", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc *= b;
            }
            acc
        });
    });
}

pub fn bench_goldilocks_add(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Goldilocks::rand(&mut rng);
    let b = Goldilocks::rand(&mut rng);

    c.bench_function("Goldilocks add", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc += b;
            }
            acc
        });
    });
}

pub fn bench_fp2_goldilocks_mul(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Fp2Goldilocks::rand(&mut rng);
    let b = Fp2Goldilocks::rand(&mut rng);

    c.bench_function("Fp2<Goldilocks> multiply", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc *= b;
            }
            acc
        });
    });
}

pub fn bench_fp2_goldilocks_base_mult_extension(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Fp2Goldilocks::rand(&mut rng);
    let b = Goldilocks::rand(&mut rng);

    c.bench_function("Fp2<Goldilocks> multiply by basefield", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc.mul_assign_by_basefield(&b);
            }
            acc
        });
    });
}

pub fn bench_fp2_goldilocks_add(c: &mut Criterion) {
    let mut rng = test_rng();
    let a = Fp2Goldilocks::rand(&mut rng);
    let b = Fp2Goldilocks::rand(&mut rng);

    c.bench_function("Fp2<Goldilocks> add", |bench| {
        bench.iter(|| {
            let mut acc = a;
            for _ in 0..1000 {
                acc += b;
            }
            acc
        });
    });
}

// smalls

// pub fn bench_small_fp4_mul(c: &mut Criterion) {
//     let mut rng = test_rng();
//     let a = Fp4SmallM31::rand(&mut rng);
//     let b = Fp4SmallM31::rand(&mut rng);

//     c.bench_function("Fp4<SmallM31> multiply", |bench| {
//         bench.iter(|| {
//             let mut acc = a;
//             // Chain 1000 multiplications to smooth noise
//             for _ in 0..1000 {
//                 acc *= b;
//             }
//             acc
//         });
//     });
// }

// pub fn bench_small_f128_mul(c: &mut Criterion) {
//     let mut rng = test_rng();
//     let a = SmallF128Mont::rand(&mut rng);
//     let b = SmallF128Mont::rand(&mut rng);

//     c.bench_function("Small F128 multiply", |bench| {
//         bench.iter(|| {
//             let mut acc = a;
//             for _ in 0..1000 {
//                 acc *= b;
//             }
//             acc
//         });
//     });
// }

// pub fn bench_small_fp2_goldilocks_mul(c: &mut Criterion) {
//     let mut rng = test_rng();
//     let a = Fp2SmallGoldilocks::rand(&mut rng);
//     let b = Fp2SmallGoldilocks::rand(&mut rng);

//     c.bench_function("Fp2<SmallGoldilocks> multiply", |bench| {
//         bench.iter(|| {
//             let mut acc = a;
//             for _ in 0..1000 {
//                 acc *= b;
//             }
//             acc
//         });
//     });
// }

criterion_group!(
    benches,
    // bench_small_f128_mul,
    // bench_small_fp4_mul,
    // bench_small_fp2_goldilocks_mul,
    bench_m31_mul,
    bench_m31_add,
    bench_fp4_mul,
    bench_fp4_base_mult_extension,
    bench_goldilocks_mult,
    bench_goldilocks_add,
    bench_fp2_goldilocks_mul,
    bench_fp2_goldilocks_add,
    bench_fp2_goldilocks_base_mult_extension,
    bench_f128_add,
    bench_f128_mul,
);
criterion_main!(benches);
