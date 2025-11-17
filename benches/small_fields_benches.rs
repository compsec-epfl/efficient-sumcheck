use ark_std::{hint::black_box, test_rng, UniformRand};

use criterion::{criterion_group, criterion_main, Criterion};
use efficient_sumcheck::{
    tests::{SmallF16, SmallM31},
    wip::{f16::mul_assign_16_bit_vectorized, m31::mul_assign_m31_vectorized},
};

fn bench_pairwise_mul_16_bit_prime(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<SmallF16> = (0..1 << 20).map(|_| SmallF16::rand(&mut rng)).collect();
    let b: Vec<SmallF16> = (0..1 << 20).map(|_| SmallF16::rand(&mut rng)).collect();

    c.bench_function("pairwise_mul_16_bit_prime", |bencher| {
        bencher.iter(|| {
            for i in 0..a.len() {
                a[i] *= &b[i];
            }
            black_box(&a);
        });
    });
}

fn bench_pairwise_mul_16_bit_prime_vectorized(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<SmallF16> = (0..1 << 20).map(|_| SmallF16::rand(&mut rng)).collect();
    let b: Vec<SmallF16> = (0..1 << 20).map(|_| SmallF16::rand(&mut rng)).collect();

    let a_raw: &mut [u16] =
        unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u16, a.len()) };
    let b_raw: &[u16] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const u16, b.len()) };

    c.bench_function("pairwise_mul_16_bit_prime_vectorized", |bencher| {
        bencher.iter(|| {
            mul_assign_16_bit_vectorized(a_raw, b_raw);
            black_box(&a);
        });
    });
}

fn bench_pairwise_mul_m31(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<SmallM31> = (0..1 << 20).map(|_| SmallM31::rand(&mut rng)).collect();
    let b: Vec<SmallM31> = (0..1 << 20).map(|_| SmallM31::rand(&mut rng)).collect();

    c.bench_function("pairwise_mul_m31", |bencher| {
        bencher.iter(|| {
            for i in 0..a.len() {
                a[i] *= &b[i];
            }
            black_box(&a);
        });
    });
}

fn bench_pairwise_mul_m31_vectorized(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<SmallM31> = (0..1 << 20).map(|_| SmallM31::rand(&mut rng)).collect();
    let b: Vec<SmallM31> = (0..1 << 20).map(|_| SmallM31::rand(&mut rng)).collect();

    let a_raw: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u32, a.len()) };
    let b_raw: &[u32] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const u32, b.len()) };

    c.bench_function("pairwise_mul_m31_vectorized", |bencher| {
        bencher.iter(|| {
            mul_assign_m31_vectorized(a_raw, b_raw);
            black_box(&a);
        });
    });
}

criterion_group!(
    benches,
    bench_pairwise_mul_16_bit_prime,
    bench_pairwise_mul_16_bit_prime_vectorized,
    bench_pairwise_mul_m31,
    bench_pairwise_mul_m31_vectorized,
);
criterion_main!(benches);
