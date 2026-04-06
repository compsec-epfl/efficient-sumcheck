use ark_ff::UniformRand;
use ark_std::{hint::black_box, time::Duration};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};

use efficient_sumcheck::{
    multilinear_sumcheck,
    simd_fields::{goldilocks::GoldilocksNeon, SimdBaseField},
    tests::F64,
    transcript::SanityTranscript,
};

fn get_bench_group(c: &mut Criterion) -> BenchmarkGroup<'_, WallTime> {
    let mut group = c.benchmark_group("simd_vs_generic");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    group
}

fn simd_vs_generic_sumcheck(c: &mut Criterion) {
    let mut group = get_bench_group(c);

    for num_vars in [16, 17, 18, 19, 20, 24] {
        let n = 1usize << num_vars;

        // ── Generic multilinear_sumcheck (auto-dispatches to SIMD for F64) ──
        group.bench_with_input(
            BenchmarkId::new("generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        evals
                    },
                    |mut evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(multilinear_sumcheck::<F64, F64>(
                            &mut evals,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        // ── Raw SIMD (no conversion — simulates SmallFp / zero-cost transmute) ──
        group.bench_with_input(
            BenchmarkId::new("simd_raw", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let evals: Vec<u64> =
                            (0..n).map(|_| F64::rand(&mut rng).value).collect();
                        evals
                    },
                    |evals| {
                        use efficient_sumcheck::simd_sumcheck::prove::prove_base_eq_ext;
                        let mut challenge_idx = 0u64;
                        black_box(prove_base_eq_ext::<GoldilocksNeon>(
                            &evals,
                            |_s0, _s1| {
                                challenge_idx = challenge_idx
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1);
                                challenge_idx % GoldilocksNeon::MODULUS
                            },
                        ));
                    },
                )
            },
        );

        // ── Generic sumcheck with same fixed challenges (apples-to-apples) ──
        group.bench_with_input(
            BenchmarkId::new("generic_fixed_chg", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        evals
                    },
                    |mut evals| {
                        use efficient_sumcheck::multilinear::reductions::pairwise;
                        let num_rounds = evals.len().trailing_zeros() as usize;
                        let mut msgs = Vec::with_capacity(num_rounds);
                        let mut challenge_idx = 0u64;
                        for _ in 0..num_rounds {
                            let msg = pairwise::evaluate(&evals);
                            msgs.push(msg);
                            challenge_idx = challenge_idx
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1);
                            let chg =
                                F64::from(challenge_idx % GoldilocksNeon::MODULUS);
                            pairwise::reduce_evaluations(&mut evals, chg);
                        }
                        black_box(msgs);
                    },
                )
            },
        );
    }

    group.finish();
}

criterion_group!(benches, simd_vs_generic_sumcheck);
criterion_main!(benches);
