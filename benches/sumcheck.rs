//! Sumcheck benchmarks for the canonical API.
//!
//! Matrix: {multilinear, inner_product} × {F64, F64Ext3} × {2^16, 2^20, 2^24}
//! Plus: fold kernel throughput.
//!
//! Run:   cargo bench --bench sumcheck
//! AVX:   RUSTFLAGS="-C target-feature=+avx512ifma" cargo bench --bench sumcheck

use ark_ff::UniformRand;
use ark_std::{hint::black_box, time::Duration};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use effsc::provers::inner_product::InnerProductProver;
use effsc::provers::multilinear::MultilinearProver;
use effsc::runner::sumcheck;
use effsc::tests::{F64Ext3, F64};
use effsc::transcript::SanityTranscript;

const SIZES: [usize; 3] = [16, 20, 24];

fn multilinear_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("multilinear/F64");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    for nv in SIZES {
        let n = 1usize << nv;
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("2^{nv}")), |b| {
            b.iter_with_setup(
                || {
                    let mut rng = ark_std::test_rng();
                    (0..n).map(|_| F64::rand(&mut rng)).collect::<Vec<_>>()
                },
                |evals| {
                    let mut p = MultilinearProver::new(evals);
                    let mut rng = ark_std::test_rng();
                    let mut t = SanityTranscript::new(&mut rng);
                    black_box(sumcheck(&mut p, nv, &mut t, |_, _| {}));
                },
            );
        });
    }
    g.finish();
}

fn multilinear_ext3(c: &mut Criterion) {
    let mut g = c.benchmark_group("multilinear/F64Ext3");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    for nv in SIZES {
        let n = 1usize << nv;
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("2^{nv}")), |b| {
            b.iter_with_setup(
                || {
                    let mut rng = ark_std::test_rng();
                    (0..n).map(|_| F64Ext3::rand(&mut rng)).collect::<Vec<_>>()
                },
                |evals| {
                    let mut p = MultilinearProver::new(evals);
                    let mut rng = ark_std::test_rng();
                    let mut t = SanityTranscript::new(&mut rng);
                    black_box(sumcheck(&mut p, nv, &mut t, |_, _| {}));
                },
            );
        });
    }
    g.finish();
}

fn inner_product_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("inner_product/F64");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    for nv in SIZES {
        let n = 1usize << nv;
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("2^{nv}")), |b| {
            b.iter_with_setup(
                || {
                    let mut rng = ark_std::test_rng();
                    let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                    let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                    (a, b)
                },
                |(a, b)| {
                    let mut p = InnerProductProver::new(a, b);
                    let mut rng = ark_std::test_rng();
                    let mut t = SanityTranscript::new(&mut rng);
                    black_box(sumcheck(&mut p, nv, &mut t, |_, _| {}));
                },
            );
        });
    }
    g.finish();
}

fn inner_product_ext3(c: &mut Criterion) {
    let mut g = c.benchmark_group("inner_product/F64Ext3");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    for nv in SIZES {
        let n = 1usize << nv;
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("2^{nv}")), |b| {
            b.iter_with_setup(
                || {
                    let mut rng = ark_std::test_rng();
                    let a: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                    let b: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                    (a, b)
                },
                |(a, b)| {
                    let mut p = InnerProductProver::new(a, b);
                    let mut rng = ark_std::test_rng();
                    let mut t = SanityTranscript::new(&mut rng);
                    black_box(sumcheck(&mut p, nv, &mut t, |_, _| {}));
                },
            );
        });
    }
    g.finish();
}

fn fold_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("fold/F64");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    for nv in SIZES {
        let n = 1usize << nv;
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("2^{nv}")), |b| {
            b.iter_with_setup(
                || {
                    let mut rng = ark_std::test_rng();
                    let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                    let w = F64::rand(&mut rng);
                    (evals, w)
                },
                |(mut evals, w)| {
                    effsc::fold(&mut evals, w);
                    black_box(evals);
                },
            );
        });
    }
    g.finish();
}

fn fold_ext3(c: &mut Criterion) {
    let mut g = c.benchmark_group("fold/F64Ext3");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    for nv in SIZES {
        let n = 1usize << nv;
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::from_parameter(format!("2^{nv}")), |b| {
            b.iter_with_setup(
                || {
                    let mut rng = ark_std::test_rng();
                    let evals: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                    let w = F64Ext3::rand(&mut rng);
                    (evals, w)
                },
                |(mut evals, w)| {
                    effsc::fold(&mut evals, w);
                    black_box(evals);
                },
            );
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    multilinear_f64,
    multilinear_ext3,
    inner_product_f64,
    inner_product_ext3,
    fold_f64,
    fold_ext3,
);
criterion_main!(benches);
