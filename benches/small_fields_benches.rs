use ark_std::{hint::black_box, test_rng, UniformRand};

use criterion::{criterion_group, criterion_main, Criterion};
use efficient_sumcheck::{
    multilinear::{pairwise, ReduceMode, TimeProver},
    prover::Prover,
    tests::{BenchStream, Fp4SmallM31, SmallM31, F128},
    wip::{
        fiat_shamir::BenchFiatShamir,
        m31::{
            evaluate_bf::evaluate_bf, evaluate_ef::evaluate_ef, reduce_bf::reduce_bf,
            reduce_ef::reduce_ef, sumcheck,
        },
    },
    Sumcheck,
};

pub fn bench_sumcheck_time(c: &mut Criterion) {
    const NUM_VARIABLES: usize = 18;

    let evaluation_stream: BenchStream<F128> = BenchStream::new(NUM_VARIABLES);
    let claim = evaluation_stream.claimed_sum;

    c.bench_function("sumcheck::time_prover_smallm31_2^20", |b| {
        b.iter(|| {
            // Fresh prover + RNG each iteration to simulate a full run
            let mut time_prover = TimeProver::<F128, BenchStream<F128>>::new(<TimeProver<
                F128,
                BenchStream<F128>,
            > as Prover<F128>>::ProverConfig::new(
                claim,
                NUM_VARIABLES,
                evaluation_stream.clone(),
                ReduceMode::Pairwise,
            ));

            let mut rng = test_rng();
            let transcript = Sumcheck::<F128>::prove::<
                BenchStream<F128>,
                TimeProver<F128, BenchStream<F128>>,
            >(&mut time_prover, &mut rng);

            black_box(transcript);
        });
    });

    let len = 1 << NUM_VARIABLES;
    let evals: Vec<SmallM31> = (0..len).map(|x| SmallM31::from(x as u32)).collect();

    c.bench_function("sumcheck::fp4_smallm31_2^20", |b| {
        b.iter(|| {
            let mut fs = BenchFiatShamir::<Fp4SmallM31, _>::new(test_rng());
            let transcript = sumcheck::prove(&evals, &mut fs);
            black_box(transcript);
        });
    });
}

fn bench_reduce_ef(c: &mut Criterion) {
    const LEN_XSMALL: usize = 1 << 10; // 1K
    const LEN_SMALL: usize = 1 << 14; // 16K
    const LEN_MED: usize = 1 << 16; // 64K
    const LEN_LARGE: usize = 1 << 18; // 256K
    const LEN_XLARGE: usize = 1 << 20; // 1M

    let mut rng = test_rng();

    // Shared input vector in the base field
    let src_xsmall: Vec<Fp4SmallM31> = (0..LEN_XSMALL)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();
    let src_small: Vec<Fp4SmallM31> = (0..LEN_SMALL)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();
    let src_med: Vec<Fp4SmallM31> = (0..LEN_MED).map(|_| Fp4SmallM31::rand(&mut rng)).collect();
    let src_large: Vec<Fp4SmallM31> = (0..LEN_LARGE)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();
    let src_xlarge: Vec<Fp4SmallM31> = (0..LEN_XLARGE)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();

    let challenge_ef = Fp4SmallM31::from(7);

    // This should be faster
    c.bench_function("reduce_ef::reduce_1K", |b| {
        b.iter(|| {
            let mut v = src_xsmall.clone();
            reduce_ef(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("reduce_ef::reduce_16K", |b| {
        b.iter(|| {
            let mut v = src_small.clone();
            reduce_ef(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("reduce_ef::reduce_64K", |b| {
        b.iter(|| {
            let mut v = src_med.clone();
            reduce_ef(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("reduce_ef::reduce_256K", |b| {
        b.iter(|| {
            let mut v = src_large.clone();
            reduce_ef(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("reduce_ef::reduce_1M", |b| {
        b.iter(|| {
            let mut v = src_xlarge.clone();
            reduce_ef(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("ef_pairwise::reduce_1K", |b| {
        b.iter(|| {
            let mut v = src_xsmall.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("ef_pairwise::reduce_16K", |b| {
        b.iter(|| {
            let mut v = src_small.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("ef_pairwise::reduce_64K", |b| {
        b.iter(|| {
            let mut v = src_med.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("ef_pairwise::reduce_256K", |b| {
        b.iter(|| {
            let mut v = src_large.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_ef);
        });
    });

    c.bench_function("ef_pairwise::reduce_1M", |b| {
        b.iter(|| {
            let mut v = src_xlarge.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_ef);
        });
    });
}

fn bench_reduce_bf(c: &mut Criterion) {
    const LEN_XSMALL: usize = 1 << 10; // 1K
    const LEN_SMALL: usize = 1 << 14; // 16K
    const LEN_MED: usize = 1 << 16; // 64K
    const LEN_LARGE: usize = 1 << 18; // 256K
    const LEN_XLARGE: usize = 1 << 20; // 1M

    let mut rng = test_rng();

    // Shared input vector in the base field
    let src_xsmall: Vec<SmallM31> = (0..LEN_XSMALL).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_small: Vec<SmallM31> = (0..LEN_SMALL).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_med: Vec<SmallM31> = (0..LEN_MED).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_large: Vec<SmallM31> = (0..LEN_LARGE).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_xlarge: Vec<SmallM31> = (0..LEN_XLARGE).map(|_| SmallM31::rand(&mut rng)).collect();

    let challenge_ef = Fp4SmallM31::from(7);

    let src_xsmall_f128: Vec<F128> = (0..LEN_XSMALL).map(|_| F128::rand(&mut rng)).collect();
    let src_small_f128: Vec<F128> = (0..LEN_SMALL).map(|_| F128::rand(&mut rng)).collect();
    let src_med_f128: Vec<F128> = (0..LEN_MED).map(|_| F128::rand(&mut rng)).collect();
    let src_large_f128: Vec<F128> = (0..LEN_LARGE).map(|_| F128::rand(&mut rng)).collect();
    let src_xlarge_f128: Vec<F128> = (0..LEN_XLARGE).map(|_| F128::rand(&mut rng)).collect();

    let challenge_f128 = F128::from(7);

    // This should be faster
    c.bench_function("reduce_bf::reduce_1K", |b| {
        b.iter(|| {
            let v = src_xsmall.clone();
            reduce_bf(black_box(&v), challenge_ef);
        });
    });

    c.bench_function("reduce_bf::reduce_16K", |b| {
        b.iter(|| {
            let v = src_small.clone();
            reduce_bf(black_box(&v), challenge_ef);
        });
    });

    c.bench_function("reduce_bf::reduce_64K", |b| {
        b.iter(|| {
            let v = src_med.clone();
            reduce_bf(black_box(&v), challenge_ef);
        });
    });

    c.bench_function("reduce_bf::reduce_256K", |b| {
        b.iter(|| {
            let v = src_large.clone();
            reduce_bf(black_box(&v), challenge_ef);
        });
    });

    c.bench_function("reduce_bf::reduce_1M", |b| {
        b.iter(|| {
            let v = src_xlarge.clone();
            reduce_bf(black_box(&v), challenge_ef);
        });
    });

    c.bench_function("bf_pairwise::reduce_1K", |b| {
        b.iter(|| {
            let mut v = src_xsmall_f128.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_f128);
        });
    });

    c.bench_function("bf_pairwise::reduce_16K", |b| {
        b.iter(|| {
            let mut v = src_small_f128.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_f128);
        });
    });

    c.bench_function("bf_pairwise::reduce_64K", |b| {
        b.iter(|| {
            let mut v = src_med_f128.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_f128);
        });
    });

    c.bench_function("bf_pairwise::reduce_256K", |b| {
        b.iter(|| {
            let mut v = src_large_f128.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_f128);
        });
    });

    c.bench_function("bf_pairwise::reduce_1M", |b| {
        b.iter(|| {
            let mut v = src_xlarge_f128.clone();
            pairwise::reduce_evaluations(black_box(&mut v), challenge_f128);
        });
    });
}

fn bench_evaluate_bf(c: &mut Criterion) {
    const LEN_XSMALL: usize = 1 << 10; // 1K
    const LEN_SMALL: usize = 1 << 14; // 16K
    const LEN_MED: usize = 1 << 16; // 64K
    const LEN_LARGE: usize = 1 << 18; // 256K
    const LEN_XLARGE: usize = 1 << 20; // 1M

    let mut rng = test_rng();

    // Shared input vector in the base field
    let src_xsmall: Vec<SmallM31> = (0..LEN_XSMALL).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_small: Vec<SmallM31> = (0..LEN_SMALL).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_med: Vec<SmallM31> = (0..LEN_MED).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_large: Vec<SmallM31> = (0..LEN_LARGE).map(|_| SmallM31::rand(&mut rng)).collect();
    let src_xlarge: Vec<SmallM31> = (0..LEN_XLARGE).map(|_| SmallM31::rand(&mut rng)).collect();

    let src_xsmall_f128: Vec<F128> = (0..LEN_XSMALL).map(|_| F128::rand(&mut rng)).collect();
    let src_small_f128: Vec<F128> = (0..LEN_SMALL).map(|_| F128::rand(&mut rng)).collect();
    let src_med_f128: Vec<F128> = (0..LEN_MED).map(|_| F128::rand(&mut rng)).collect();
    let src_large_f128: Vec<F128> = (0..LEN_LARGE).map(|_| F128::rand(&mut rng)).collect();
    let src_xlarge_f128: Vec<F128> = (0..LEN_XLARGE).map(|_| F128::rand(&mut rng)).collect();

    // This should be faster
    c.bench_function("evaluate_bf::evaluate_1K", |b| {
        b.iter(|| {
            let v = src_xsmall.clone();
            evaluate_bf::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_bf::evaluate_16K", |b| {
        b.iter(|| {
            let v = src_small.clone();
            evaluate_bf::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_bf::evaluate_64K", |b| {
        b.iter(|| {
            let v = src_med.clone();
            evaluate_bf::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_bf::evaluate_256K", |b| {
        b.iter(|| {
            let v = src_large.clone();
            evaluate_bf::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_bf::evaluate_1M", |b| {
        b.iter(|| {
            let v = src_xlarge.clone();
            evaluate_bf::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("bf_pairwise::evaluate_1K", |b| {
        b.iter(|| {
            let v = src_xsmall_f128.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("bf_pairwise::evaluate_16K", |b| {
        b.iter(|| {
            let v = src_small_f128.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("bf_pairwise::evaluate_64K", |b| {
        b.iter(|| {
            let v = src_med_f128.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("bf_pairwise::evaluate_256K", |b| {
        b.iter(|| {
            let v = src_large_f128.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("bf_pairwise::evaluate_1M", |b| {
        b.iter(|| {
            let v = src_xlarge_f128.clone();
            pairwise::evaluate(black_box(&v));
        });
    });
}

fn bench_evaluate_ef(c: &mut Criterion) {
    const LEN_XSMALL: usize = 1 << 10; // 1K
    const LEN_SMALL: usize = 1 << 14; // 16K
    const LEN_MED: usize = 1 << 16; // 64K
    const LEN_LARGE: usize = 1 << 18; // 256K
    const LEN_XLARGE: usize = 1 << 20; // 1M

    let mut rng = test_rng();

    // Shared input vector in the base field
    let src_xsmall: Vec<Fp4SmallM31> = (0..LEN_XSMALL)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();
    let src_small: Vec<Fp4SmallM31> = (0..LEN_SMALL)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();
    let src_med: Vec<Fp4SmallM31> = (0..LEN_MED).map(|_| Fp4SmallM31::rand(&mut rng)).collect();
    let src_large: Vec<Fp4SmallM31> = (0..LEN_LARGE)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();
    let src_xlarge: Vec<Fp4SmallM31> = (0..LEN_XLARGE)
        .map(|_| Fp4SmallM31::rand(&mut rng))
        .collect();

    // This should be faster
    c.bench_function("evaluate_ef::evaluate_1K", |b| {
        b.iter(|| {
            let v = src_xsmall.clone();
            evaluate_ef::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_ef::evaluate_16K", |b| {
        b.iter(|| {
            let v = src_small.clone();
            evaluate_ef::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_ef::evaluate_64K", |b| {
        b.iter(|| {
            let v = src_med.clone();
            evaluate_ef::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_ef::evaluate_256K", |b| {
        b.iter(|| {
            let v = src_large.clone();
            evaluate_ef::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("evaluate_ef::evaluate_1M", |b| {
        b.iter(|| {
            let v = src_xlarge.clone();
            evaluate_ef::<2_147_483_647>(black_box(&v));
        });
    });

    c.bench_function("pairwise::evaluate_1K", |b| {
        b.iter(|| {
            let v = src_xsmall.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("pairwise::evaluate_16K", |b| {
        b.iter(|| {
            let v = src_small.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("pairwise::evaluate_64K", |b| {
        b.iter(|| {
            let v = src_med.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("pairwise::evaluate_256K", |b| {
        b.iter(|| {
            let v = src_large.clone();
            pairwise::evaluate(black_box(&v));
        });
    });

    c.bench_function("pairwise::evaluate_1M", |b| {
        b.iter(|| {
            let v = src_xlarge.clone();
            pairwise::evaluate(black_box(&v));
        });
    });
}

criterion_group!(
    benches,
    bench_sumcheck_time,
    bench_reduce_ef,
    bench_reduce_bf,
    bench_evaluate_ef,
    bench_evaluate_bf,
);
criterion_main!(benches);
