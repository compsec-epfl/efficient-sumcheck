use ark_ff::Field;
use ark_poly::multivariate::{SparsePolynomial, SparseTerm};

use crate::{
    prover::{ProductProverConfig, Prover},
    streams::{multivariate_product_claim, Stream},
    tests::{
        multilinear_product::provers::basic::{BasicProductProver, BasicProductProverConfig},
        polynomials::Polynomial,
        BenchStream,
    },
    ProductSumcheck,
};

pub fn consistency_test<F, S, P>()
where
    F: Field,
    S: Stream<F> + From<BenchStream<F>> + Clone,
    P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F, F)>>,
    P::ProverConfig: ProductProverConfig<F, S>,
{
    // get a stream
    let num_variables = 16;
    let s: S = BenchStream::new(num_variables).into();
    let claim = multivariate_product_claim(vec![s.clone(), s.clone()]);

    // get the sanity prover
    let s_evaluations: Vec<F> = (0..1 << num_variables).map(|i| s.evaluation(i)).collect();
    let p: SparsePolynomial<F, SparseTerm> =
        <SparsePolynomial<F, SparseTerm> as Polynomial<F>>::from_hypercube_evaluations(
            s_evaluations.clone(),
        );
    let mut sanity_prover = BasicProductProver::<F>::new(BasicProductProverConfig::new(
        claim,
        num_variables,
        p.clone(),
        p,
    ));

    // prove
    let prover_transcript = ProductSumcheck::<F>::prove::<BenchStream<F>, P>(
        &mut P::new(ProductProverConfig::default(
            num_variables,
            vec![s.clone(), s],
        )),
        &mut ark_std::test_rng(),
    );

    let sanity_prover_transcript = ProductSumcheck::<F>::prove::<
        BenchStream<F>,
        BasicProductProver<F>,
    >(&mut sanity_prover, &mut ark_std::test_rng());

    // ensure the transcript is identical
    assert_eq!(prover_transcript, sanity_prover_transcript);
}
