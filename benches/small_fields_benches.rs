use ark_ff::Field;
use ark_std::{hint::black_box, test_rng, UniformRand};

use criterion::{criterion_group, criterion_main, Criterion};
use efficient_sumcheck::{
    multilinear::pairwise,
    tests::{Fp4SmallM31, SmallF16, SmallM31},
    wip::{
        f16::mul_assign_16_bit_vectorized,
        m31::{
            mul_assign_m31_vectorized,
            vectorized_reductions::{self, pairwise::reduce_evaluations_bf},
        },
    },
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

fn bench_pairwise_evaluate(c: &mut Criterion) {
    const LEN_SMALL: usize = 1 << 10; // 1K
    const LEN_MED: usize = 1 << 16; // 64K
    const LEN_LARGE: usize = 1 << 20; // 1M

    let mut rng = test_rng();

    // Prepare inputs once and reuse them in all iterations
    let src_small: Vec<SmallM31> = (0..LEN_SMALL).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_med: Vec<SmallM31> = (0..LEN_MED).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_large: Vec<SmallM31> = (0..LEN_LARGE).map(|_| SmallM31::rand(&mut rng)).collect();

    // ----- SIMD evaluate -----
    c.bench_function("vectorized::pairwise::evaluate_1K", |b| {
        b.iter(|| vectorized_reductions::pairwise::evaluate_bf(black_box(&src_small)));
    });

    c.bench_function("vectorized::pairwise::evaluate_64K", |b| {
        b.iter(|| vectorized_reductions::pairwise::evaluate_bf(black_box(&src_med)));
    });

    c.bench_function("vectorized::pairwise::evaluate_1M", |b| {
        b.iter(|| vectorized_reductions::pairwise::evaluate_bf(black_box(&src_large)));
    });

    // ----- scalar (pairwise) -----
    c.bench_function("pairwise::evaluate_1K", |b| {
        b.iter(|| pairwise::evaluate(black_box(&src_small)));
    });

    c.bench_function("pairwise::evaluate_64K", |b| {
        b.iter(|| pairwise::evaluate(black_box(&src_med)));
    });

    c.bench_function("pairwise::evaluate_1M", |b| {
        b.iter(|| pairwise::evaluate(black_box(&src_large)));
    });
}

fn bench_reduce_evaluations_bf(c: &mut Criterion) {
    const LEN: usize = 1 << 20;
    let mut rng = test_rng();

    // Shared input vector in the base field
    let src: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();

    let challenge = SmallM31::from(7u32);
    let challenge_ext = Fp4SmallM31::from_base_prime_field(challenge);

    // 2) New: direct extension-field reduce_evaluations_bf
    c.bench_function("reduce_evaluations_bf", |b| {
        b.iter(|| {
            let out_ext = reduce_evaluations_bf(black_box(&src), black_box(challenge_ext));

            black_box(out_ext);
        });
    });

    // 1) Baseline: base-field reduce_evaluations + lift to Fp4SmallM31
    c.bench_function("reduce_evaluations", |b| {
        b.iter(|| {
            // pairwise::reduce_evaluations mutates & shrinks the vector,
            // so we clone src each iteration for a fair benchmark.
            let mut src_copy = src.clone();

            pairwise::reduce_evaluations(
                black_box(&mut src_copy),
                black_box(challenge),
            );

            black_box(src_copy);
        });
    });
}

criterion_group!(
    benches,
    bench_reduce_evaluations_bf,
    bench_pairwise_evaluate,
    bench_pairwise_mul_16_bit_prime,
    bench_pairwise_mul_16_bit_prime_vectorized,
    bench_pairwise_mul_m31,
    bench_pairwise_mul_m31_vectorized,
);
criterion_main!(benches);
