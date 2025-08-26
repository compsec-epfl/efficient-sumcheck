use ark_std::{hint::black_box, time::Duration};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use space_efficient_sumcheck::{
    multilinear::TimeProver,
    multilinear_product::TimeProductProver,
    prover::{ProductProverConfig, Prover, ProverConfig},
    streams::multivariate_product_claim,
    tests::{BenchStream, F128},
    ProductSumcheck, Sumcheck,
};

fn time_prover_bench(c: &mut Criterion) {
    // Build a custom config so it runs long enough to measure
    let mut g = c.benchmark_group("sumcheck_prover");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10));

    // If creating the stream is expensive, you can create once and clone in setup.
    let n_vars = 24usize;

    g.bench_function("time_prover", |b| {
        b.iter_batched(
            // --- setup (NOT timed) ---
            || {
                // Fresh stream + claim each iteration (or clone)
                let evaluation_stream = BenchStream::<F128>::new(n_vars);
                let claim = evaluation_stream.claimed_sum;

                // Fresh prover each iteration (or provide a reset() method on it)
                let prover = TimeProver::<F128, BenchStream<F128>>::new(<TimeProver<
                    F128,
                    BenchStream<F128>,
                > as Prover<F128>>::ProverConfig::default(
                    claim,
                    n_vars,
                    evaluation_stream,
                ));

                // Fresh RNG each iteration
                let rng = ark_std::test_rng();

                // Pass everything into the timed closure
                (prover, rng)
            },
            // --- measurement (timed) ---
            |(mut prover, mut rng)| {
                let proof = Sumcheck::<F128>::prove::<
                    BenchStream<F128>,
                    TimeProver<F128, BenchStream<F128>>,
                >(&mut prover, &mut rng);
                black_box(proof);
            },
            BatchSize::SmallInput, // or LargeInput if setup is cheap
        )
    });

    g.finish();
}

pub fn time_product_prover_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("sumcheck_prover");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10));

    let n_vars = 24usize;

    g.bench_function("time_product_prover", |b| {
        b.iter_batched(
            // ------------ setup (not timed) ------------
            || {
                // Build one stream; clone into the config (cheap if BenchStream is Clone-by-arc)
                let evaluation_stream: BenchStream<F128> = BenchStream::<F128>::new(n_vars);
                let streams: Vec<BenchStream<F128>> =
                    vec![evaluation_stream.clone(), evaluation_stream.clone()];
                let claim = multivariate_product_claim(streams.clone());

                // Prover from config
                let prover: TimeProductProver<F128, BenchStream<F128>> =
                    TimeProductProver::<F128, BenchStream<F128>>::new(
                        ProductProverConfig::default(claim, n_vars, streams),
                    );

                // Fresh RNG each iter
                let rng = ark_std::test_rng();

                (prover, rng)
            },
            // ------------ measurement (timed) ------------
            |(mut prover, mut rng)| {
                // Let inference figure out the generic params of `prove`
                let proof: ProductSumcheck<F128> = ProductSumcheck::prove::<
                    BenchStream<F128>,
                    TimeProductProver<F128, BenchStream<F128>>,
                >(&mut prover, &mut rng);
                black_box(proof);
            },
            BatchSize::SmallInput, // bump to LargeInput if setup is tiny vs proving
        )
    });

    g.finish();
}

criterion_group!(benches, time_product_prover_bench, time_prover_bench);
criterion_main!(benches);
