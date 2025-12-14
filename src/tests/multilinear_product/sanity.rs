use ark_ff::Field;

use crate::{
    multilinear::ReduceMode,
    multilinear_product::TimeProductProverConfig,
    prover::{ProductProverConfig, Prover},
    streams::{MemoryStream, Stream},
    tests::polynomials::four_variable_polynomial_evaluations,
};

fn multilinear_product_round_sanity<F, P>(
    round_num: usize,
    p: &mut P,
    message: Option<F>,
    eval_0: F,
    eval_1: F,
) where
    F: Field,
    P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F, F)>>,
{
    let round = p.next_message(message).unwrap();
    assert_eq!(
        round.0, eval_0,
        "g0 should evaluate correctly round {}",
        round_num
    );
    assert_eq!(
        round.1, eval_1,
        "g1 should evaluate correctly round {}",
        round_num
    );
}

pub fn sanity_test_driver<F, P>(p: &mut P)
where
    F: Field,
    P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F, F)>>,
{
    /*
     * Zeroth Round:
     *
     * Evaluations:
     *   0000 →  0 * 0  =  0
     *   0001 →  1 * 1  =  1
     *   0010 →  0 * 0  =  0
     *   0011 →  1 * 1  =  1
     *   0100 → 13 * 13 = 17
     *   0101 → 14 * 14 =  6
     *   0110 →  1 * 1  =  1
     *   0111 →  2 * 2  =  4
     *   ----------------------
     *   Sum g₀(0) = 11
     *
     *   1000 →  2 * 2  =  4
     *   1001 →  3 * 3  =  9
     *   1010 →  2 * 2  =  4
     *   1011 →  3 * 3  =  9
     *   1100 →  0 * 0  =  0
     *   1101 →  1 * 1  =  1
     *   1110 →  7 * 7  = 11
     *   1111 →  8 * 8  =  7
     *   ----------------------
     *   Sum g₀(1) = 7
     */
    multilinear_product_round_sanity::<F, P>(0, p, None, F::from(11_u32), F::from(7_u32));
    /*
     * First Round: x₀ fixed to 3
     *
     * Evaluations for g₁(0):
     *   3000 → (0 * 17) + (2 * 3) = 6 * 6  = 17
     *   3001 → (1 * 17) + (3 * 3) = 7 * 7  = 11
     *   3010 → (0 * 17) + (2 * 3) = 6 * 6  = 17
     *   3011 → (1 * 17) + (3 * 3) = 7 * 7  = 11
     *   -------------------------------
     *   Sum g₁(0) = 18
     *
     * Evaluations for g₁(1):
     *   3100 → (13 * 17) + (0 * 3) = 12 * 12 = 11
     *   3101 → (14 * 17) + (1 * 3) = 13 * 13 = 17
     *   3110 → (1 * 17)  + (7 * 3) = 0 * 0   = 0
     *   3111 → (2 * 17)  + (8 * 3) = 1 * 1   = 1
     *   -------------------------------
     *   Sum g₁(1) = 10
     */
    multilinear_product_round_sanity::<F, P>(
        1,
        p,
        Some(F::from(3_u32)),
        F::from(18_u32),
        F::from(10_u32),
    );
    /*
     * Second Round: x₁ fixed to 4
     *
     * Evaluations for g₂(0):
     *   3400 → (6 * 16) + (12 * 4) = 11 * 11 = 7
     *   3401 → (7 * 16) + (13 * 4) = 12 * 12 = 11
     *   -------------------------------
     *   Sum g₂(0) = 18
     *
     * Evaluations for g₂(1):
     *   3410 → (6 * 16) + (0 * 4) = 1 * 1 = 1
     *   3411 → (7 * 16) + (1 * 4) = 2 * 2 = 4
     *   -------------------------------
     *   Sum g₂(1) = 5
     */
    multilinear_product_round_sanity::<F, P>(
        2,
        p,
        Some(F::from(4_u32)),
        F::from(18_u32),
        F::from(5_u32),
    );
    /*
     * Last Round: x₂ fixed to 7
     *
     * Evaluations for g₃(0):
     *   3470 → (11 * 13) + (1 * 7) = 17 * 17 = 4
     *   -------------------------------
     *   Sum g₃(0) = 4
     *
     * Evaluations for g₃(1):
     *   3471 → (12 * 13) + (2 * 7) = 18 * 18 = 1
     *   -------------------------------
     *   Sum g₃(1) = 1
     */
    multilinear_product_round_sanity::<F, P>(
        3,
        p,
        Some(F::from(7_u32)),
        F::from(4_u32),
        F::from(1_u32),
    );
}

pub fn sanity_test<F, S, P>()
where
    F: Field,
    S: Stream<F> + From<MemoryStream<F>>,
    P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F, F)>>,
    P::ProverConfig: ProductProverConfig<F, S>,
{
    let s_p: S = MemoryStream::new(four_variable_polynomial_evaluations()).into();
    let s_q: S = MemoryStream::new(four_variable_polynomial_evaluations()).into();
    // let c = P::ProverConfig::new(4, vec![s_p, s_q], ReduceMode::Variablewise);
    let mut p = P::new(ProductProverConfig::default(4, vec![s_p, s_q]));
    sanity_test_driver(&mut p);
}

// pub fn pairwise_sanity_test<F, S, P>()
// where
//     F: Field,
//     S: Stream<F> + From<MemoryStream<F>>,
//     P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F, F)>>,
//     P::ProverConfig: ProductProverConfig<F, S>,
// {
//     let s_p: S = MemoryStream::new(four_variable_polynomial_evaluations()).into();
//     let s_q: S = MemoryStream::new(four_variable_polynomial_evaluations()).into();
//     let config = TimeProductProverConfig::new(4, vec![s_p, s_q], ReduceMode::Pairwise);
//     let mut p = P::new(TimeProductProverConfig::new(4, vec![s_p, s_q], ReduceMode::Pairwise));
//     sanity_test_driver(&mut p);
// }
