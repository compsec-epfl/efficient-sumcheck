use ark_ff::Field;
use criterion::{criterion_group, criterion_main, Criterion};
use space_efficient_sumcheck::tests::{SmallF32, SmallF32Mont};
use std::hint::black_box;

fn bench_mul_standard(c: &mut Criterion) {
    c.bench_function("SmallField_mul", |b| {
        b.iter(|| {
            let mut v = SmallF32::ONE;
            let v2 = SmallF32::new(20);
            for _ in 0..1_000_000 {
                v *= v2;
            }
            black_box(v);
        })
    });
}

fn bench_mul_montgomery(c: &mut Criterion) {
    c.bench_function("SmallFieldMont_mul", |b| {
        b.iter(|| {
            let mut v = SmallF32Mont::ONE;
            let v2 = SmallF32Mont::new(20);
            for _ in 0..1_000_000 {
                v *= v2;
            }
            black_box(v);
        })
    });
}

criterion_group!(benches, bench_mul_standard, bench_mul_montgomery);
criterion_main!(benches);
