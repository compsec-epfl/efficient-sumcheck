use ark_std::{hint::black_box, time::Duration};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, Criterion,
};

use efficient_sumcheck::{
    multilinear::TimeProver,
    multilinear_product::TimeProductProver,
    prover::{ProductProverConfig, Prover, ProverConfig},
    tests::{BenchStream, F128},
    ProductSumcheck, Sumcheck,
};

fn get_bench_group(c: &mut Criterion) -> BenchmarkGroup<'_, WallTime> {
    let mut group = c.benchmark_group("sumcheck_prover");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10));
    group
}

fn time_prover_bench(c: &mut Criterion) {
    let num_vars = 24usize;
    get_bench_group(c).bench_function("time_prover", |bencher| {
        bencher.iter_batched(
            || {
                let stream = BenchStream::<F128>::new(num_vars);
                TimeProver::<F128, BenchStream<F128>>::new(
                    <TimeProver<F128, BenchStream<F128>> as Prover<F128>>::ProverConfig::default(
                        num_vars, stream,
                    ),
                )
            },
            |mut prover: TimeProver<F128, BenchStream<F128>>| {
                black_box(Sumcheck::<F128>::prove::<
                    BenchStream<F128>,
                    TimeProver<F128, BenchStream<F128>>,
                >(&mut prover, &mut ark_std::test_rng()));
            },
            BatchSize::LargeInput,
        )
    });
}

fn time_product_prover_bench(c: &mut Criterion) {
    let num_vars = 24usize;
    get_bench_group(c).bench_function("time_product_prover", |bencher| {
        bencher.iter_batched(
            || {
                let stream = BenchStream::<F128>::new(num_vars);
                let streams: Vec<BenchStream<F128>> = vec![stream.clone(), stream.clone()];
                TimeProductProver::<F128, BenchStream<F128>>::new(ProductProverConfig::default(
                    num_vars, streams,
                ))
            },
            |mut prover: TimeProductProver<F128, BenchStream<F128>>| {
                black_box(ProductSumcheck::<F128>::prove::<
                    BenchStream<F128>,
                    TimeProductProver<F128, BenchStream<F128>>,
                >(&mut prover, &mut ark_std::test_rng()));
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, time_product_prover_bench, time_prover_bench);
criterion_main!(benches);
