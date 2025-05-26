use ark_bn254::Fr as BN254Field;
use ark_ff::Field;

use space_efficient_sumcheck::{
    hypercube::Hypercube,
    multilinear::{
        BlendyProver, BlendyProverConfig, SpaceProver, SpaceProverConfig, TimeProver,
        TimeProverConfig,
    },
    multilinear_product::{
        BlendyProductProver, BlendyProductProverConfig, TimeProductProver, TimeProductProverConfig, SpaceProductProver,
        SpaceProductProverConfig,
    },
    order_strategy::SignificantBitOrder,
    prover::{Prover, ProverConfig},
    streams::{multivariate_claim, multivariate_product_claim, MemoryStream},
    tests::{F128, F64},
    ProductSumcheck, Sumcheck,
};

pub mod validation;
use validation::{validate_and_format_command_line_args, AlgorithmLabel, BenchArgs, FieldLabel};

fn run_on_field<F: Field>(bench_args: BenchArgs) {
    let mut rng = ark_std::test_rng();

    // create a MemoryStream
    let mut evaluations = Vec::with_capacity(Hypercube::<SignificantBitOrder>::stop_value(bench_args.num_variables));
    for i in 0..Hypercube::<SignificantBitOrder>::stop_value(bench_args.num_variables) {
        evaluations.push(F::from(i as u64));
    }
    let s = MemoryStream::<F>::new(evaluations);

    // switch on algorithm_label
    match bench_args.algorithm_label {
        AlgorithmLabel::Blendy => {
            let config: BlendyProverConfig<F, MemoryStream<F>> =
                BlendyProverConfig::<F, MemoryStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let transcript =
                Sumcheck::<F>::prove::<MemoryStream<F>, BlendyProver<F, MemoryStream<F>>>(
                    &mut BlendyProver::<F, MemoryStream<F>>::new(config),
                    &mut rng,
                );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::VSBW => {
            let config: TimeProverConfig<F, MemoryStream<F>> =
                TimeProverConfig::<F, MemoryStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let transcript = Sumcheck::<F>::prove::<MemoryStream<F>, TimeProver<F, MemoryStream<F>>>(
                &mut TimeProver::<F, MemoryStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::CTY => {
            let config: SpaceProverConfig<F, MemoryStream<F>> =
                SpaceProverConfig::<F, MemoryStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let transcript = Sumcheck::<F>::prove::<MemoryStream<F>, SpaceProver<F, MemoryStream<F>>>(
                &mut SpaceProver::<F, MemoryStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::ProductVSBW => {
            let config: TimeProductProverConfig<F, MemoryStream<F>> =
                TimeProductProverConfig::<F, MemoryStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(), s.clone()]),
                    num_variables: bench_args.num_variables,
                    streams: vec![s.clone(), s],
                };
            let transcript = ProductSumcheck::<F>::prove::<
                MemoryStream<F>,
                TimeProductProver<F, MemoryStream<F>>,
            >(
                &mut TimeProductProver::<F, MemoryStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::ProductBlendy => {
            let config: BlendyProductProverConfig<F, MemoryStream<F>> =
                BlendyProductProverConfig::<F, MemoryStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(), s.clone()]),
                    num_variables: bench_args.num_variables,
                    num_stages: bench_args.stage_size,
                    streams: vec![s.clone(), s],
                };
            let transcript = ProductSumcheck::<F>::prove::<
                MemoryStream<F>,
                BlendyProductProver<F, MemoryStream<F>>,
            >(
                &mut BlendyProductProver::<F, MemoryStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::ProductCTY => {
            let config: SpaceProductProverConfig<F, MemoryStream<F>> =
                SpaceProductProverConfig::<F, MemoryStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(), s.clone()]),
                    num_variables: bench_args.num_variables,
                    streams: vec![s.clone(), s],
                };
            let transcript = ProductSumcheck::<F>::prove::<
                MemoryStream<F>,
                SpaceProductProver<F, MemoryStream<F>>,
            >(
                &mut SpaceProductProver::<F, MemoryStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
    };
}

fn main() {
    // Collect command line arguments
    let bench_args: BenchArgs = validate_and_format_command_line_args(std::env::args().collect());
    // Run the requested bench
    match bench_args.field_label {
        FieldLabel::Field64 => {
            run_on_field::<F64>(bench_args);
        }
        FieldLabel::Field128 => {
            run_on_field::<F128>(bench_args);
        }
        FieldLabel::FieldBn254 => {
            run_on_field::<BN254Field>(bench_args);
        }
    };
}
