use ark_ff::UniformRand;
use ark_std::{hint::black_box, time::Duration};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};

use efficient_sumcheck::{
    inner_product_sumcheck,
    multilinear::reductions::pairwise,
    multilinear_sumcheck,
    tests::{F64Ext2, F64Ext3, F64},
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

    for num_vars in [16, 18, 20, 22, 24] {
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

    for num_vars in [16, 18, 20, 22, 24] {
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

    for num_vars in [16, 18, 20, 22, 24] {
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

    for num_vars in [16, 18, 20, 22, 24] {
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
                            len = reduce::reduce_in_place::<SimdBackend>(&mut current[..len], *chg);
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
                            len = reduce::reduce_in_place::<SimdBackend>(&mut current[..len], *chg);
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

// ── Inner product sumcheck ──────────────────────────────────────────────────

fn inner_product_sumcheck_bench(c: &mut Criterion) {
    use efficient_sumcheck::inner_product_sumcheck;

    let mut group = c.benchmark_group("inner_product_sumcheck");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for num_vars in [16, 18, 20, 22, 24] {
        let n = 1usize << num_vars;

        // ── Auto-dispatch (SIMD for Goldilocks) ──
        group.bench_with_input(
            BenchmarkId::new("auto_dispatch", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        let g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(mut f, mut g)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(inner_product_sumcheck::<F64, F64>(
                            &mut f,
                            &mut g,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        // ── Generic path with same transcript overhead ──
        group.bench_with_input(
            BenchmarkId::new("generic_pairwise", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        let g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(f, g)| {
                        use efficient_sumcheck::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;

                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = f.len().trailing_zeros() as usize;
                        let mut prover_msgs = Vec::with_capacity(num_rounds);

                        // Round 0 in BF
                        let msg = pairwise_product_evaluate(&[f.clone(), g.clone()]);
                        prover_msgs.push(msg);
                        transcript.write(msg.0);
                        transcript.write(msg.1);
                        let chg: F64 = transcript.read();
                        let mut ef_f = pairwise::cross_field_reduce(&f, chg);
                        let mut ef_g = pairwise::cross_field_reduce(&g, chg);

                        // Rounds 1+
                        for _ in 1..num_rounds {
                            let msg = pairwise_product_evaluate(&[ef_f.clone(), ef_g.clone()]);
                            prover_msgs.push(msg);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64 = transcript.read();
                            pairwise::reduce_evaluations(&mut ef_f, chg);
                            pairwise::reduce_evaluations(&mut ef_g, chg);
                        }
                        black_box(prover_msgs);
                    },
                )
            },
        );
    }

    group.finish();
}

// ── Coefficient sumcheck ────────────────────────────────────────────────────

fn coefficient_sumcheck_bench(c: &mut Criterion) {
    use efficient_sumcheck::coefficient_sumcheck::{coefficient_sumcheck, RoundPolyEvaluator};

    struct Degree1Eval;
    impl RoundPolyEvaluator<F64> for Degree1Eval {
        fn degree(&self) -> usize {
            1
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (even, odd) = pw[0];
            coeffs[0] += even;
            coeffs[1] += odd - even;
        }
    }

    struct MixedEval;
    impl RoundPolyEvaluator<F64> for MixedEval {
        fn degree(&self) -> usize {
            0
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            coeffs[0] += tw[0].0[0] + pw[0].0;
        }
    }

    let mut group = c.benchmark_group("coefficient_sumcheck");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for num_vars in [16, 18, 20, 22, 24] {
        let n = 1usize << num_vars;

        // ── Pairwise reduce only (isolate reduce cost) ──
        group.bench_with_input(
            BenchmarkId::new("reduce_only", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n).map(|_| F64::rand(&mut rng)).collect::<Vec<F64>>()
                    },
                    |evals| {
                        let mut pw = vec![evals];
                        let num_rounds = pw[0].len().trailing_zeros() as usize;
                        let chg = F64::from(7u64);
                        for _ in 0..num_rounds {
                            pairwise::reduce_evaluations(&mut pw[0], chg);
                        }
                        black_box(pw);
                    },
                )
            },
        );

        // ── Degree-1: evaluator trait (parallel + SIMD reduce) ──
        group.bench_with_input(
            BenchmarkId::new("degree1_auto", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n).map(|_| F64::rand(&mut rng)).collect::<Vec<F64>>()
                    },
                    |evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let mut pw = vec![evals];
                        let mut tw: Vec<Vec<Vec<F64>>> = vec![];
                        black_box(coefficient_sumcheck(
                            &Degree1Eval,
                            &mut tw,
                            &mut pw,
                            num_vars,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        // ── Degree-1: generic (manual reduce, no SIMD) ──
        group.bench_with_input(
            BenchmarkId::new("degree1_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n).map(|_| F64::rand(&mut rng)).collect::<Vec<F64>>()
                    },
                    |evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let mut pw = vec![evals];
                        let num_rounds = pw[0].len().trailing_zeros() as usize;
                        let mut msgs = Vec::with_capacity(num_rounds);
                        for _ in 0..num_rounds {
                            let s0: F64 = pw[0].iter().step_by(2).copied().sum();
                            let s1: F64 = pw[0].iter().skip(1).step_by(2).copied().sum();
                            transcript.write(s0);
                            let c: F64 = transcript.read();
                            msgs.push((s0, s1));
                            pairwise::reduce_evaluations(&mut pw[0], c);
                        }
                        black_box(msgs);
                    },
                )
            },
        );

        // ── Tablewise 2-col: evaluator trait ──
        group.bench_with_input(
            BenchmarkId::new("tablewise_auto", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let table: Vec<Vec<F64>> = (0..n)
                            .map(|_| vec![F64::rand(&mut rng), F64::rand(&mut rng)])
                            .collect();
                        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        (table, evals)
                    },
                    |(table, evals)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let mut tw = vec![table];
                        let mut pw = vec![evals];
                        black_box(coefficient_sumcheck(
                            &MixedEval,
                            &mut tw,
                            &mut pw,
                            num_vars,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        // ── Tablewise 2-col: generic (no SIMD) ──
        group.bench_with_input(
            BenchmarkId::new("tablewise_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                use efficient_sumcheck::multilinear::reductions::tablewise;
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let table: Vec<Vec<F64>> = (0..n)
                            .map(|_| vec![F64::rand(&mut rng), F64::rand(&mut rng)])
                            .collect();
                        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        (table, evals)
                    },
                    |(table, evals)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let mut tw = vec![table];
                        let mut pw = vec![evals];
                        let num_rounds = pw[0].len().trailing_zeros() as usize;
                        let mut msgs = Vec::with_capacity(num_rounds);
                        for _ in 0..num_rounds {
                            let ts: F64 = tw[0].iter().map(|row| row[0]).sum();
                            let ps: F64 = pw[0].iter().step_by(2).copied().sum();
                            transcript.write(ts + ps);
                            let c: F64 = transcript.read();
                            msgs.push(ts + ps);
                            tablewise::reduce_evaluations(&mut tw[0], c);
                            pairwise::reduce_evaluations(&mut pw[0], c);
                        }
                        black_box(msgs);
                    },
                )
            },
        );
    }

    group.finish();
}

// ── Extension field sumcheck ────────────────────────────────────────────────

fn extension_field_sumcheck_bench(c: &mut Criterion) {
    use efficient_sumcheck::tests::F64Ext2;

    let mut group = c.benchmark_group("extension_sumcheck");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for num_vars in [16, 18, 20, 22, 24] {
        let n = 1usize << num_vars;

        // ── F64Ext2 (degree-2 extension, SIMD ext evaluate dispatched) ──
        group.bench_with_input(
            BenchmarkId::new("ext2_auto", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n)
                            .map(|_| F64Ext2::rand(&mut rng))
                            .collect::<Vec<F64Ext2>>()
                    },
                    |mut evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(multilinear_sumcheck::<F64Ext2, F64Ext2>(
                            &mut evals,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        // ── F64Ext2 generic (no SIMD evaluate) ──
        group.bench_with_input(
            BenchmarkId::new("ext2_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n)
                            .map(|_| F64Ext2::rand(&mut rng))
                            .collect::<Vec<F64Ext2>>()
                    },
                    |evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = evals.len().trailing_zeros() as usize;
                        let mut ef_evals = evals;
                        let mut msgs = Vec::with_capacity(num_rounds);
                        for _ in 0..num_rounds {
                            let msg = pairwise::evaluate(&ef_evals);
                            msgs.push(msg);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64Ext2 = transcript.read();
                            pairwise::reduce_evaluations(&mut ef_evals, chg);
                        }
                        black_box(msgs);
                    },
                )
            },
        );

        // ── F64Ext3 (degree-3 extension, SIMD ext evaluate dispatched) ──
        group.bench_with_input(
            BenchmarkId::new("ext3_auto", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n)
                            .map(|_| F64Ext3::rand(&mut rng))
                            .collect::<Vec<F64Ext3>>()
                    },
                    |mut evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(multilinear_sumcheck::<F64Ext3, F64Ext3>(
                            &mut evals,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        // ── F64Ext3 generic ──
        group.bench_with_input(
            BenchmarkId::new("ext3_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        (0..n)
                            .map(|_| F64Ext3::rand(&mut rng))
                            .collect::<Vec<F64Ext3>>()
                    },
                    |evals| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = evals.len().trailing_zeros() as usize;
                        let mut ef_evals = evals;
                        let mut msgs = Vec::with_capacity(num_rounds);
                        for _ in 0..num_rounds {
                            let msg = pairwise::evaluate(&ef_evals);
                            msgs.push(msg);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64Ext3 = transcript.read();
                            pairwise::reduce_evaluations(&mut ef_evals, chg);
                        }
                        black_box(msgs);
                    },
                )
            },
        );
    }

    group.finish();
}

// ── Inner product with extension fields ─────────────────────────────────────

fn inner_product_extension_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("ip_extension");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for num_vars in [16, 18, 20, 22, 24] {
        let n = 1usize << num_vars;

        group.bench_with_input(
            BenchmarkId::new("ext2", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
                        let g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(mut f, mut g)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(inner_product_sumcheck::<F64Ext2, F64Ext2>(
                            &mut f,
                            &mut g,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ext2_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
                        let g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(f, g)| {
                        use efficient_sumcheck::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;

                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = f.len().trailing_zeros() as usize;
                        let mut ef_f = f;
                        let mut ef_g = g;
                        for _ in 0..num_rounds {
                            let msg = pairwise_product_evaluate(&[ef_f.clone(), ef_g.clone()]);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64Ext2 = transcript.read();
                            pairwise::reduce_evaluations(&mut ef_f, chg);
                            pairwise::reduce_evaluations(&mut ef_g, chg);
                        }
                        black_box((ef_f, ef_g));
                    },
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ext3", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        let g: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(mut f, mut g)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(inner_product_sumcheck::<F64Ext3, F64Ext3>(
                            &mut f,
                            &mut g,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ext3_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        let g: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(f, g)| {
                        use efficient_sumcheck::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;

                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = f.len().trailing_zeros() as usize;
                        let mut ef_f = f;
                        let mut ef_g = g;
                        for _ in 0..num_rounds {
                            let msg = pairwise_product_evaluate(&[ef_f.clone(), ef_g.clone()]);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64Ext3 = transcript.read();
                            pairwise::reduce_evaluations(&mut ef_f, chg);
                            pairwise::reduce_evaluations(&mut ef_g, chg);
                        }
                        black_box((ef_f, ef_g));
                    },
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("base", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        let g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(mut f, mut g)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(inner_product_sumcheck::<F64, F64>(
                            &mut f,
                            &mut g,
                            &mut transcript,
                        ));
                    },
                )
            },
        );

        // ── Generic baselines (no simd_ops, raw arkworks) ──
        group.bench_with_input(
            BenchmarkId::new("ext2_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                use efficient_sumcheck::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate_slices;
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
                        let g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(f, g)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = f.len().trailing_zeros() as usize;
                        let mut ef_f = f;
                        let mut ef_g = g;
                        let mut msgs = Vec::with_capacity(num_rounds);
                        for _ in 0..num_rounds {
                            let msg = pairwise_product_evaluate_slices(&ef_f, &ef_g);
                            msgs.push(msg);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64Ext2 = transcript.read();
                            pairwise::reduce_evaluations(&mut ef_f, chg);
                            pairwise::reduce_evaluations(&mut ef_g, chg);
                        }
                        black_box(msgs);
                    },
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ext3_generic", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                use efficient_sumcheck::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate_slices;
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        let g: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(f, g)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        let num_rounds = f.len().trailing_zeros() as usize;
                        let mut ef_f = f;
                        let mut ef_g = g;
                        let mut msgs = Vec::with_capacity(num_rounds);
                        for _ in 0..num_rounds {
                            let msg = pairwise_product_evaluate_slices(&ef_f, &ef_g);
                            msgs.push(msg);
                            transcript.write(msg.0);
                            transcript.write(msg.1);
                            let chg: F64Ext3 = transcript.read();
                            pairwise::reduce_evaluations(&mut ef_f, chg);
                            pairwise::reduce_evaluations(&mut ef_g, chg);
                        }
                        black_box(msgs);
                    },
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ext3", format!("2^{}", num_vars)),
            &num_vars,
            |bencher, _| {
                bencher.iter_with_setup(
                    || {
                        let mut rng = ark_std::test_rng();
                        let f: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        let g: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
                        (f, g)
                    },
                    |(mut f, mut g)| {
                        let mut rng = ark_std::test_rng();
                        let mut transcript = SanityTranscript::new(&mut rng);
                        black_box(inner_product_sumcheck::<F64Ext3, F64Ext3>(
                            &mut f,
                            &mut g,
                            &mut transcript,
                        ));
                    },
                )
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
    bench_eval_reduce_loop,
    inner_product_sumcheck_bench,
    coefficient_sumcheck_bench,
    extension_field_sumcheck_bench,
    inner_product_extension_bench
);
criterion_main!(benches);
