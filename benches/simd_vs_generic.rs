use ark_ff::UniformRand;
use ark_std::{hint::black_box, time::Duration};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};

use efficient_sumcheck::{
    multilinear::reductions::pairwise,
    multilinear_sumcheck,
    tests::F64,
    transcript::{SanityTranscript, Transcript},
};

fn get_bench_group(c: &mut Criterion) -> BenchmarkGroup<'_, WallTime> {
    let mut group = c.benchmark_group("simd_vs_generic");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    group
}

/// End-to-end sumcheck: SIMD auto-dispatch vs generic pairwise.
///
/// Both paths use the same SanityTranscript for apples-to-apples comparison.
/// The "auto_dispatch" path goes through `multilinear_sumcheck` which detects
/// Goldilocks and routes to SIMD. The "generic" path calls pairwise
/// evaluate/reduce directly with the same transcript overhead.
fn simd_vs_generic_sumcheck(c: &mut Criterion) {
    let mut group = get_bench_group(c);

    for num_vars in [16, 18, 20, 24] {
        let n = 1usize << num_vars;

        // ── multilinear_sumcheck (auto-dispatches to SIMD for Goldilocks) ──
        group.bench_with_input(
            BenchmarkId::new("auto_dispatch", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n).map(|_| F64::rand(&mut rng)).collect::<Vec<F64>>()
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

        // ── Generic pairwise with same SanityTranscript overhead ──
        group.bench_with_input(
            BenchmarkId::new("generic_pairwise", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n).map(|_| F64::rand(&mut rng)).collect::<Vec<F64>>()
                    },
                    |mut evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = evals.len().trailing_zeros() as usize;
                        let mut prover_msgs = Vec::with_capacity(num_rounds);
                        for _ in 0..num_rounds {
                            let msg = pairwise::evaluate(&evals);
                            prover_msgs.push(msg);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64 = transcript.read();
                            pairwise::reduce_evaluations(&mut evals, chg);
                        }
                        black_box(prover_msgs);
                    },
                )
            },
        );
    }

    group.finish();
}

// ── Isolated evaluate micro-benchmarks ──────────────────────────────────────

fn bench_evaluate_isolated(c: &mut Criterion) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    use efficient_sumcheck::simd_fields::goldilocks::GoldilocksAvx512 as SimdBackend;
    #[cfg(target_arch = "aarch64")]
    use efficient_sumcheck::simd_fields::goldilocks::GoldilocksNeon as SimdBackend;
    use efficient_sumcheck::simd_sumcheck::evaluate;

    let mut group = c.benchmark_group("evaluate_isolated");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));

    for num_vars in [16, 20, 24] {
        let n = 1usize << num_vars;

        group.bench_with_input(
            BenchmarkId::new("simd", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                let mut rng = ark_std::test_rng();
                let evals: Vec<u64> = (0..n).map(|_| F64::rand(&mut rng).value).collect();
                bencher.iter(|| {
                    black_box(evaluate::evaluate_parallel::<SimdBackend>(&evals));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                let mut rng = ark_std::test_rng();
                let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                bencher.iter(|| {
                    black_box(pairwise::evaluate(&evals));
                });
            },
        );
    }

    group.finish();
}

// ── Isolated reduce micro-benchmarks ────────────────────────────────────────

fn bench_reduce_isolated(c: &mut Criterion) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    use efficient_sumcheck::simd_fields::goldilocks::GoldilocksAvx512 as SimdBackend;
    #[cfg(target_arch = "aarch64")]
    use efficient_sumcheck::simd_fields::goldilocks::GoldilocksNeon as SimdBackend;
    use efficient_sumcheck::simd_sumcheck::reduce;

    let mut group = c.benchmark_group("reduce_isolated");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));

    for num_vars in [16, 20, 24] {
        let n = 1usize << num_vars;

        group.bench_with_input(
            BenchmarkId::new("simd_parallel", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                let mut rng = ark_std::test_rng();
                let evals: Vec<u64> = (0..n).map(|_| F64::rand(&mut rng).value).collect();
                let challenge = F64::rand(&mut rng).value;
                bencher.iter(|| {
                    black_box(reduce::reduce_parallel::<SimdBackend>(&evals, challenge));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_in_place", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                let mut rng = ark_std::test_rng();
                let evals: Vec<u64> = (0..n).map(|_| F64::rand(&mut rng).value).collect();
                let challenge = F64::rand(&mut rng).value;
                bencher.iter_with_setup(
                    || evals.clone(),
                    |mut e| {
                        black_box(reduce::reduce_in_place::<SimdBackend>(&mut e, challenge));
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                let mut rng = ark_std::test_rng();
                let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                let challenge = F64::rand(&mut rng);
                bencher.iter_with_setup(
                    || evals.clone(),
                    |mut e| {
                        pairwise::reduce_evaluations(&mut e, challenge);
                        black_box(e);
                    },
                );
            },
        );
    }

    group.finish();
}

// ── Eval+Reduce loop (no transcript overhead) ───────────────────────────────

fn bench_eval_reduce_loop(c: &mut Criterion) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    use efficient_sumcheck::simd_fields::goldilocks::GoldilocksAvx512 as SimdBackend;
    #[cfg(target_arch = "aarch64")]
    use efficient_sumcheck::simd_fields::goldilocks::GoldilocksNeon as SimdBackend;
    use efficient_sumcheck::simd_sumcheck::{evaluate, reduce};

    let mut group = c.benchmark_group("eval_reduce_loop");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for num_vars in [16, 20, 24] {
        let n = 1usize << num_vars;

        // Minimal loop with per-round random challenge (no copy overhead)
        group.bench_with_input(
            BenchmarkId::new("simd_loop", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let evals: Vec<u64> = (0..n).map(|_| F64::rand(&mut rng).value).collect();
                        let challenges: Vec<u64> =
                            (0..num_vars).map(|_| F64::rand(&mut rng).value).collect();
                        (evals, challenges)
                    },
                    |(mut current, challenges)| {
                        let mut len = current.len();
                        for chg in &challenges {
                            let _ = evaluate::evaluate_parallel::<SimdBackend>(&current[..len]);
                            len = reduce::reduce_in_place::<SimdBackend>(
                                &mut current[..len],
                                *chg,
                            );
                        }
                        black_box(current);
                    },
                );
            },
        );

        // Copy moved to setup (isolates compute from allocation)
        group.bench_with_input(
            BenchmarkId::new("simd_dispatch_like", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        let buf: &[u64] = unsafe {
                            core::slice::from_raw_parts(evals.as_ptr() as *const u64, evals.len())
                        };
                        let current = buf.to_vec();
                        let challenges: Vec<u64> =
                            (0..num_vars).map(|_| F64::rand(&mut rng).value).collect();
                        (current, challenges)
                    },
                    |(mut current, challenges)| {
                        let mut len = current.len();
                        for chg in &challenges {
                            let (s0, s1) =
                                evaluate::evaluate_parallel::<SimdBackend>(&current[..len]);
                            black_box((s0, s1));
                            len = reduce::reduce_in_place::<SimdBackend>(
                                &mut current[..len],
                                *chg,
                            );
                        }
                        black_box(current);
                    },
                );
            },
        );

        // Fused: reduce + next evaluate in a single pass
        group.bench_with_input(
            BenchmarkId::new("simd_fused", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let evals: Vec<u64> = (0..n).map(|_| F64::rand(&mut rng).value).collect();
                        let challenges: Vec<u64> =
                            (0..num_vars).map(|_| F64::rand(&mut rng).value).collect();
                        (evals, challenges)
                    },
                    |(mut current, challenges)| {
                        let mut len = current.len();
                        // First evaluate standalone
                        let (mut s0, mut s1) =
                            evaluate::evaluate_parallel::<SimdBackend>(&current[..len]);
                        for (round, chg) in challenges.iter().enumerate() {
                            black_box((s0, s1));
                            if round < num_vars - 1 {
                                // Fused reduce + next evaluate
                                let (ns0, ns1, new_len) =
                                    reduce::reduce_and_evaluate::<SimdBackend>(
                                        &mut current[..len],
                                        *chg,
                                    );
                                len = new_len;
                                s0 = ns0;
                                s1 = ns1;
                            }
                        }
                        black_box(current);
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("generic_loop", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        let challenge = F64::rand(&mut rng);
                        (evals, challenge)
                    },
                    |(mut evals, challenge)| {
                        for _ in 0..num_vars {
                            let _ = pairwise::evaluate(&evals);
                            pairwise::reduce_evaluations(&mut evals, challenge);
                        }
                        black_box(evals);
                    },
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    simd_vs_generic_sumcheck,
    bench_evaluate_isolated,
    bench_reduce_isolated,
    bench_eval_reduce_loop
);
criterion_main!(benches);
