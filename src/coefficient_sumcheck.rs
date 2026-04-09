use ark_ff::Field;
use ark_poly::univariate::DensePolynomial;
use ark_poly::Polynomial;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::multilinear::reductions::{pairwise, tablewise};
use crate::transcript::Transcript;

#[derive(Debug)]
pub struct CoefficientSumcheck<F: Field> {
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
pub trait RoundPolyEvaluator<F: Field>: Sync {
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
pub fn coefficient_sumcheck<F: Field>(
    evaluator: &impl RoundPolyEvaluator<F>,
    tablewise: &mut [Vec<Vec<F>>],
    pairwise: &mut [Vec<F>],
    n_rounds: usize,
    transcript: &mut impl Transcript<F>,
) -> CoefficientSumcheck<F> {
    let mut prover_messages = Vec::with_capacity(n_rounds);
    let mut verifier_messages = Vec::with_capacity(n_rounds);

    let n_tw = tablewise.len();
    let n_pw = pairwise.len();
    let deg = evaluator.degree();
    let n_coeffs = deg + 1;

    for _ in 0..n_rounds {
        let n_pairs = if n_tw > 0 {
            tablewise[0].len() / 2
        } else if n_pw > 0 {
            pairwise[0].len() / 2
        } else {
            0
        };

        // Accumulate round polynomial coefficients.
        // Each pair adds its contribution into a coefficient buffer.
        // For rayon: each thread gets its own buffer, summed at the end.
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

        #[cfg(feature = "parallel")]
        let coeffs = (0..n_pairs)
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
            .unwrap_or_else(|| vec![F::ZERO; n_coeffs]);

        #[cfg(not(feature = "parallel"))]
        let coeffs = {
            let mut coeffs = vec![F::ZERO; n_coeffs];
            for pair_idx in 0..n_pairs {
                accumulate_at(&mut coeffs, pair_idx);
            }
            coeffs
        };

        let round_poly = DensePolynomial { coeffs };

        // Send only the first d coefficients (omit the leading one).
        // The verifier derives it from h(0) + h(1) = claim.
        let d = round_poly.coeffs.len().saturating_sub(1);
        for coeff in &round_poly.coeffs[..d] {
            transcript.write(*coeff);
        }

        prover_messages.push(round_poly);

        let c = transcript.read();
        verifier_messages.push(c);

        for table in tablewise.iter_mut() {
            tablewise::reduce_evaluations(table, c);
        }
        for table in pairwise.iter_mut() {
            #[cfg(any(
                target_arch = "aarch64",
                all(target_arch = "x86_64", target_feature = "avx512ifma")
            ))]
            if crate::simd_sumcheck::dispatch::try_simd_reduce(table, c) {
                continue;
            }
            pairwise::reduce_evaluations(table, c);
        }
    }

    CoefficientSumcheck {
        prover_messages,
        verifier_messages,
    }
}

/// Sumcheck verifier for arbitrary-degree round polynomials in coefficient form.
///
/// Each round: absorb the first `d` coefficients → derive the leading coefficient
/// from `c_d = claim - 2·c_0 - c_1 - ... - c_{d-1}` → squeeze challenge
/// → update `claim = h(challenge)`.
///
/// The prover messages contain the **full** polynomial (including the leading
/// coefficient), but only the first `d` coefficients are absorbed into the
/// transcript — matching what the prover sends.
pub fn sumcheck_verify<F: Field>(
    claim: &mut F,
    prover_messages: &[DensePolynomial<F>],
    transcript: &mut impl Transcript<F>,
) -> Option<Vec<F>> {
    let mut challenges = Vec::with_capacity(prover_messages.len());

    for h in prover_messages {
        let d = h.coeffs.len().saturating_sub(1);

        // Absorb only the first d coefficients (leading one is derived).
        for coeff in &h.coeffs[..d] {
            transcript.write(*coeff);
        }

        // Derive leading coefficient: c_d = claim - 2*c_0 - c_1 - ... - c_{d-1}
        let partial_sum: F = h.coeffs[..d].iter().skip(1).copied().sum();
        let expected_leading = *claim - h.coeffs[0].double() - partial_sum;

        // Verify the prover's leading coefficient matches
        if d < h.coeffs.len() && h.coeffs[d] != expected_leading {
            return None;
        }

        let c = transcript.read();
        *claim = h.evaluate(&c);
        challenges.push(c);
    }

    Some(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_poly::DenseUVPolynomial;
    use ark_std::test_rng;

    use crate::tests::F64;
    use crate::transcript::SanityTranscript;

    // ── Reusable evaluators for tests ───────────────────────────────────

    /// Degree-1 evaluator: h(x) = even + (odd - even) * x per pair.
    struct Degree1Evaluator;
    impl RoundPolyEvaluator<F64> for Degree1Evaluator {
        fn degree(&self) -> usize { 1 }
        fn accumulate_pair(&self, coeffs: &mut [F64], _tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            let (even, odd) = pw[0];
            coeffs[0] += even;
            coeffs[1] += odd - even;
        }
    }

    /// Degree-2 evaluator: interpolate through (0, s0), (1, s1), (2, s0+s1).
    struct Degree2Evaluator;
    impl RoundPolyEvaluator<F64> for Degree2Evaluator {
        fn degree(&self) -> usize { 2 }
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
        fn degree(&self) -> usize { 0 }
        fn accumulate_pair(&self, coeffs: &mut [F64], tw: &[(&[F64], &[F64])], pw: &[(F64, F64)]) {
            coeffs[0] += tw[0].0[0] + pw[0].0;
        }
    }

    /// Inner product evaluator: per-pair product from two pairwise tables.
    struct InnerProductEvaluator;
    impl RoundPolyEvaluator<F64> for InnerProductEvaluator {
        fn degree(&self) -> usize { 1 }
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

        let domsep = spongefish::domain_separator!("test-coefficient-sumcheck"; module_path!())
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
