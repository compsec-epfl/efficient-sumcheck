use ark_ff::Field;
use ark_poly::univariate::DensePolynomial;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::reductions::{pairwise, tablewise};
use crate::transcript::ProverTranscript;

#[derive(Debug)]
pub struct CoefficientSumcheck<
    F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
> {
    pub prover_messages: Vec<DensePolynomial<F>>,
    pub verifier_messages: Vec<F>,
}

/// Trait for computing the round polynomial from a single pair of rows.
///
/// The library iterates over pairs (even/odd rows from each table),
/// calls [`accumulate_pair`](RoundPolyEvaluator::accumulate_pair) for each,
/// which adds the pair's contribution directly into a shared coefficient buffer.
/// This avoids per-pair polynomial allocation — the library owns the buffer.
///
/// # Arguments to `accumulate_pair`
///
/// - `coeffs`: mutable slice of length [`degree`](RoundPolyEvaluator::degree)`+ 1`.
///   The evaluator **adds** its contribution into these coefficients (do NOT zero them).
/// - `tablewise_pairs`: one `(even_row, odd_row)` slice-pair per tablewise table
/// - `pairwise_pairs`: one `(even_elem, odd_elem)` pair per pairwise table
///
/// # Example
///
/// ```text
/// struct MyEvaluator;
/// impl RoundPolyEvaluator<F> for MyEvaluator {
///     fn degree(&self) -> usize { 1 }
///
///     fn accumulate_pair(
///         &self,
///         coeffs: &mut [F],
///         tw: &[(&[F], &[F])],
///         pw: &[(F, F)],
///     ) {
///         let (even, odd) = pw[0];
///         coeffs[0] += even;           // constant coefficient
///         coeffs[1] += odd - even;     // linear coefficient
///     }
/// }
/// ```
pub trait RoundPolyEvaluator<
    F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
>: Sync
{
    /// The degree of the round polynomial (number of coefficients = degree + 1).
    fn degree(&self) -> usize;

    /// Accumulate this pair's contribution into `coeffs[0..=degree]`.
    ///
    /// `coeffs` is pre-zeroed at the start of each round. The evaluator
    /// should **add** (not assign) its contribution.
    fn accumulate_pair(
        &self,
        coeffs: &mut [F],
        tablewise_pairs: &[(&[F], &[F])],
        pairwise_pairs: &[(F, F)],
    );

    /// Hint: is the per-pair work heavy enough to benefit from rayon parallelism?
    ///
    /// Return `true` for evaluators that do substantial work per pair (polynomial
    /// multiplication, R1CS evaluation, etc.). Return `false` for trivial
    /// evaluators (simple sums, single multiply) where rayon overhead dominates.
    ///
    /// Default: `true` (assume heavy — safe default since rayon's overhead is
    /// small relative to the work for most real use cases).
    fn parallelize(&self) -> bool {
        true
    }
}

// ── Evaluate strategies ─────────────────────────────────────────────────────

/// SIMD fast path for degree-1 with a single pairwise table.
///
/// Returns `[sum_even, sum_odd - sum_even]` = coefficients of `h(x) = c0 + c1*x`.
fn simd_evaluate_degree1<
    F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
>(
    pw: &[F],
) -> Vec<F> {
    // Try SIMD dispatch for Goldilocks
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx512ifma")
        )
    ))]
    {
        if let Some(coeffs) = try_simd_evaluate_degree1(pw) {
            return coeffs;
        }
    }

    // Generic fallback
    let mut s0 = F::ZERO;
    let mut s1 = F::ZERO;
    for chunk in pw.chunks_exact(2) {
        s0 += chunk[0];
        s1 += chunk[1];
    }
    vec![s0, s1 - s0]
}

/// SIMD implementation of degree-1 evaluate.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
fn try_simd_evaluate_degree1<
    F: ark_ff::Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
>(
    pw: &[F],
) -> Option<Vec<F>> {
    crate::simd_sumcheck::dispatch::try_simd_evaluate_degree1(pw)
}

/// Fused SIMD reduce + degree-1 evaluate for next round.
///
/// Returns `Some([s0, s1 - s0])` if SIMD dispatch succeeded (reduces in-place
/// and computes next round's coefficients). Returns `None` to fall back to
/// separate reduce + evaluate.
#[cfg(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
))]
fn try_simd_fused_reduce_evaluate<
    F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
>(
    pw: &mut Vec<F>,
    challenge: F,
) -> Option<Vec<F>> {
    crate::simd_sumcheck::dispatch::try_simd_fused_reduce_evaluate_degree1(pw, challenge)
}

#[cfg(not(all(
    feature = "simd",
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    )
)))]
fn try_simd_fused_reduce_evaluate<
    F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
>(
    _pw: &mut Vec<F>,
    _challenge: F,
) -> Option<Vec<F>> {
    None
}

/// Parallel evaluate using rayon (for heavy evaluators).
#[cfg(feature = "parallel")]
fn parallel_evaluate<F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    n_pairs: usize,
    n_coeffs: usize,
) -> Vec<F> {
    let accumulate_at = |coeffs: &mut [F], pair_idx: usize| {
        let mut tw_buf: [(&[F], &[F]); 16] = [(&[], &[]); 16];
        let mut pw_buf: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];
        debug_assert!(n_tw <= 16 && n_pw <= 16);
        for (i, table) in tablewise.iter().enumerate() {
            tw_buf[i] = (&table[2 * pair_idx], &table[2 * pair_idx + 1]);
        }
        for (i, table) in pairwise.iter().enumerate() {
            pw_buf[i] = (table[2 * pair_idx], table[2 * pair_idx + 1]);
        }
        evaluator.accumulate_pair(coeffs, &tw_buf[..n_tw], &pw_buf[..n_pw]);
    };

    (0..n_pairs)
        .into_par_iter()
        .fold_with(vec![F::ZERO; n_coeffs], |mut acc, pair_idx| {
            accumulate_at(&mut acc, pair_idx);
            acc
        })
        .reduce_with(|mut a, b| {
            for (ai, bi) in a.iter_mut().zip(&b) {
                *ai += *bi;
            }
            a
        })
        .unwrap_or_else(|| vec![F::ZERO; n_coeffs])
}

/// Fallback when parallel feature is disabled.
#[cfg(not(feature = "parallel"))]
fn parallel_evaluate<F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    n_pairs: usize,
    n_coeffs: usize,
) -> Vec<F> {
    let mut coeffs = vec![F::ZERO; n_coeffs];
    sequential_evaluate_into(
        evaluator,
        tablewise,
        pairwise,
        n_tw,
        n_pw,
        n_pairs,
        &mut coeffs,
    );
    coeffs
}

/// Sequential evaluate (for trivial evaluators where rayon overhead dominates).
///
/// Fills `coeffs_out` with accumulated coefficients (zeroes it first).
fn sequential_evaluate_into<
    F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &[Vec<Vec<F>>],
    pairwise: &[Vec<F>],
    n_tw: usize,
    n_pw: usize,
    n_pairs: usize,
    coeffs_out: &mut [F],
) {
    for c in coeffs_out.iter_mut() {
        *c = F::ZERO;
    }
    let mut tw_buf: [(&[F], &[F]); 16] = [(&[], &[]); 16];
    let mut pw_buf: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];
    debug_assert!(n_tw <= 16 && n_pw <= 16);

    for pair_idx in 0..n_pairs {
        for (i, table) in tablewise.iter().enumerate() {
            tw_buf[i] = (&table[2 * pair_idx], &table[2 * pair_idx + 1]);
        }
        for (i, table) in pairwise.iter().enumerate() {
            pw_buf[i] = (table[2 * pair_idx], table[2 * pair_idx + 1]);
        }
        evaluator.accumulate_pair(coeffs_out, &tw_buf[..n_tw], &pw_buf[..n_pw]);
    }
}

/// Sumcheck prover for arbitrary-degree round polynomials in coefficient form.
///
/// The user provides a [`RoundPolyEvaluator`] that computes the round polynomial
/// contribution for a single pair. The library handles:
/// - Parallel iteration over pairs (via rayon when `parallel` is enabled)
/// - Summation of per-pair polynomials
/// - Transcript interaction (d-coefficient optimization: leading coefficient omitted)
/// - SIMD-accelerated pairwise reduce (auto-dispatched for Goldilocks)
/// - Tablewise reduce
pub fn coefficient_sumcheck<
    F: Field + zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable,
>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &mut [Vec<Vec<F>>],
    pairwise: &mut [Vec<F>],
    n_rounds: usize,
    transcript: &mut impl ProverTranscript<F>,
) -> CoefficientSumcheck<F> {
    let mut prover_messages = Vec::with_capacity(n_rounds);
    let mut verifier_messages = Vec::with_capacity(n_rounds);

    let n_tw = tablewise.len();
    let n_pw = pairwise.len();
    let deg = evaluator.degree();
    let n_coeffs = deg + 1;

    let use_parallel = evaluator.parallelize();
    let is_degree1_simd_path = deg == 1 && n_pw == 1 && n_tw == 0;

    let mut pending_degree1_eval: Option<Vec<F>> = None;

    // Pre-allocate coefficient buffer — reused across rounds for sequential path.
    let mut coeffs_buf = vec![F::ZERO; n_coeffs];

    for round in 0..n_rounds {
        let n_pairs = if n_tw > 0 {
            tablewise[0].len() / 2
        } else if n_pw > 0 {
            pairwise[0].len() / 2
        } else {
            0
        };

        // ── Evaluate: build round polynomial coefficients ──
        //
        // Three strategies in order of preference:
        // 1. SIMD fast path: degree-1, single pairwise table, no tablewise →
        //    use evaluate_parallel or fused reduce+evaluate
        // 2. Parallel: heavy evaluator → rayon fold_with across pairs
        // 3. Sequential: trivial evaluator → simple loop, no rayon overhead
        let coeffs = if let Some(cached) = pending_degree1_eval.take() {
            cached
        } else if is_degree1_simd_path {
            simd_evaluate_degree1::<F>(&pairwise[0])
        } else if use_parallel {
            parallel_evaluate(
                evaluator, tablewise, pairwise, n_tw, n_pw, n_pairs, n_coeffs,
            )
        } else {
            // Fill pre-allocated buffer (no allocation), then clone the
            // small coefficient vec (d+1 elements, typically 2-3).
            sequential_evaluate_into(
                evaluator,
                tablewise,
                pairwise,
                n_tw,
                n_pw,
                n_pairs,
                &mut coeffs_buf,
            );
            coeffs_buf.clone()
        };

        let round_poly = DensePolynomial { coeffs };

        // Send only the first d coefficients (omit the leading one).
        let d = round_poly.coeffs.len().saturating_sub(1);
        for coeff in &round_poly.coeffs[..d] {
            transcript.send(*coeff);
        }

        prover_messages.push(round_poly);

        let c = transcript.challenge();
        verifier_messages.push(c);

        // ── Reduce ──
        for table in tablewise.iter_mut() {
            tablewise::reduce_evaluations(table, c);
        }

        if is_degree1_simd_path && round < n_rounds - 1 {
            // Fused reduce+evaluate: SIMD reduce in-place and compute
            // next round's (s0, s1) in one pass when possible.
            if let Some(next_coeffs) = try_simd_fused_reduce_evaluate(&mut pairwise[0], c) {
                pending_degree1_eval = Some(next_coeffs);
            } else {
                // Fallback: separate reduce
                pairwise::reduce_evaluations(&mut pairwise[0], c);
            }
        } else {
            for table in pairwise.iter_mut() {
                #[cfg(all(
                    feature = "simd",
                    any(
                        target_arch = "aarch64",
                        all(target_arch = "x86_64", target_feature = "avx512ifma")
                    )
                ))]
                if crate::simd_sumcheck::dispatch::try_simd_reduce(table, c) {
                    continue;
                }
                pairwise::reduce_evaluations(table, c);
            }
        }
    }

    CoefficientSumcheck {
        prover_messages,
        verifier_messages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_poly::Polynomial;
    use ark_std::test_rng;

    use crate::tests::F64;
    use crate::transcript::SanityTranscript;

    // ── Reusable evaluators for tests ───────────────────────────────────

    /// Degree-1 evaluator: h(x) = even + (odd - even) * x per pair.
    struct Degree1Evaluator;
    impl RoundPolyEvaluator<F64> for Degree1Evaluator {
        fn degree(&self) -> usize {
            1
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (even, odd) = pw[0];
            coeffs[0] += even;
            coeffs[1] += odd - even;
        }
    }

    /// Degree-2 evaluator: interpolate through (0, s0), (1, s1), (2, s0+s1).
    struct Degree2Evaluator;
    impl RoundPolyEvaluator<F64> for Degree2Evaluator {
        fn degree(&self) -> usize {
            2
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (s0, s1) = pw[0];
            let s2 = s0 + s1;
            coeffs[0] += s0;
            coeffs[1] += (-F64::from(3u64) * s0 + F64::from(4u64) * s1 - s2) / F64::from(2u64);
            coeffs[2] += (s0 - F64::from(2u64) * s1 + s2) / F64::from(2u64);
        }
    }

    /// Mixed evaluator: tablewise column 0 + pairwise even (degree 0).
    struct MixedEvaluator;
    impl RoundPolyEvaluator<F64> for MixedEvaluator {
        fn degree(&self) -> usize {
            0
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            coeffs[0] += tw[0].0[0] + pw[0].0;
        }
    }

    /// Inner product evaluator: per-pair product from two pairwise tables.
    struct InnerProductEvaluator;
    impl RoundPolyEvaluator<F64> for InnerProductEvaluator {
        fn degree(&self) -> usize {
            1
        }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (a_even, a_odd) = pw[0];
            let (b_even, b_odd) = pw[1];
            coeffs[0] += a_even * b_even;
            coeffs[1] += a_odd * b_odd - a_even * b_even;
        }
    }

    // ── Tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_sumcheck_relation_holds_each_round() {
        let mut rng = test_rng();
        let n = 1 << 4;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            &Degree1Evaluator,
            &mut tablewise,
            &mut pairwise,
            4,
            &mut transcript,
        );

        // round 0: h(0) + h(1) == claimed_sum
        let h0 = &result.prover_messages[0];
        assert_eq!(
            h0.evaluate(&F64::from(0u64)) + h0.evaluate(&F64::from(1u64)),
            claimed_sum
        );

        // subsequent rounds: h_i(0) + h_i(1) == h_{i-1}(challenge_{i-1})
        for i in 1..result.prover_messages.len() {
            let prev_h = &result.prover_messages[i - 1];
            let challenge = result.verifier_messages[i - 1];
            let expected = prev_h.evaluate(&challenge);

            let h_i = &result.prover_messages[i];
            let actual = h_i.evaluate(&F64::from(0u64)) + h_i.evaluate(&F64::from(1u64));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_spongefish_transcript() {
        use crate::transcript::SpongefishTranscript;

        let mut rng = test_rng();
        let n = 1 << 3;
        let num_rounds = 3;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let domsep = spongefish::domain_separator!("test-coefficient-sumcheck")
            .without_session()
            .instance(b"test");

        let prover_state = domsep.std_prover();
        let mut transcript = SpongefishTranscript::new(prover_state);

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];

        let result = coefficient_sumcheck(
            &Degree1Evaluator,
            &mut tablewise,
            &mut pairwise,
            num_rounds,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), num_rounds);
        assert_eq!(result.verifier_messages.len(), num_rounds);
    }

    #[test]
    fn test_mixed_tablewise_and_pairwise() {
        let mut rng = test_rng();
        let n = 1 << 3;

        let table: Vec<Vec<F64>> = (0..n)
            .map(|_| vec![F64::rand(&mut rng), F64::rand(&mut rng)])
            .collect();
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut tablewise = vec![table];
        let mut pairwise = vec![evals];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            &MixedEvaluator,
            &mut tablewise,
            &mut pairwise,
            3,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 3);
        assert_eq!(tablewise[0].len(), 1);
        assert_eq!(pairwise[0].len(), 1);
    }

    #[test]
    fn test_higher_degree_round_polys() {
        let mut rng = test_rng();
        let n = 1 << 3;
        let evals: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let claimed_sum: F64 = evals.iter().copied().sum();

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            &Degree2Evaluator,
            &mut tablewise,
            &mut pairwise,
            3,
            &mut transcript,
        );

        let h0 = &result.prover_messages[0];
        assert_eq!(
            h0.evaluate(&F64::from(0u64)) + h0.evaluate(&F64::from(1u64)),
            claimed_sum
        );

        for h in &result.prover_messages {
            assert_eq!(h.coeffs.len(), 3);
        }
    }

    #[test]
    fn test_single_round() {
        let mut rng = test_rng();
        let evals = vec![F64::rand(&mut rng), F64::rand(&mut rng)];
        let claimed_sum = evals[0] + evals[1];

        let mut pairwise = vec![evals];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            &Degree1Evaluator,
            &mut tablewise,
            &mut pairwise,
            1,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 1);
        assert_eq!(result.verifier_messages.len(), 1);
        assert_eq!(pairwise[0].len(), 1);

        let h = &result.prover_messages[0];
        assert_eq!(
            h.evaluate(&F64::from(0u64)) + h.evaluate(&F64::from(1u64)),
            claimed_sum
        );
    }

    #[test]
    fn test_multiple_pairwise_tables() {
        let mut rng = test_rng();
        let n = 1 << 3;
        let evals_a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut pairwise = vec![evals_a, evals_b];
        let mut tablewise: Vec<Vec<Vec<F64>>> = vec![];
        let mut transcript = SanityTranscript::new(&mut rng);

        let result = coefficient_sumcheck(
            &InnerProductEvaluator,
            &mut tablewise,
            &mut pairwise,
            3,
            &mut transcript,
        );

        assert_eq!(result.prover_messages.len(), 3);
        assert_eq!(pairwise[0].len(), 1);
        assert_eq!(pairwise[1].len(), 1);
    }
}
