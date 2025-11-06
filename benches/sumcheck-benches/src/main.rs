use ark_bn254::Fr as BN254Field;
use ark_ff::Field;

use efficient_sumcheck::{
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
    streams::{multivariate_claim, multivariate_product_claim},
    tests::{BenchStream, F128, F64}, // SmallGoldilocks as F64
    ProductSumcheck, Sumcheck,
};

pub mod validation;
use validation::{validate_and_format_command_line_args, AlgorithmLabel, BenchArgs, FieldLabel};

fn run_on_field<F: Field>(bench_args: BenchArgs) {
    let mut rng = ark_std::test_rng();
    let s = BenchStream::<F>::new(bench_args.num_variables);

    // switch on algorithm_label
    match bench_args.algorithm_label {
        AlgorithmLabel::Blendy => {
            let config: BlendyProverConfig<F, BenchStream<F>> =
                BlendyProverConfig::<F, BenchStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let transcript =
                Sumcheck::<F>::prove::<BenchStream<F>, BlendyProver<F, BenchStream<F>>>(
                    &mut BlendyProver::<F, BenchStream<F>>::new(config),
                    &mut rng,
                );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::VSBW => {
            let config: TimeProverConfig<F, BenchStream<F>> =
                TimeProverConfig::<F, BenchStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let transcript = Sumcheck::<F>::prove::<BenchStream<F>, TimeProver<F, BenchStream<F>>>(
                &mut TimeProver::<F, BenchStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::CTY => {
            let config: SpaceProverConfig<F, BenchStream<F>> =
                SpaceProverConfig::<F, BenchStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let transcript = Sumcheck::<F>::prove::<BenchStream<F>, SpaceProver<F, BenchStream<F>>>(
                &mut SpaceProver::<F, BenchStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::ProductVSBW => {
            let config: TimeProductProverConfig<F, BenchStream<F>> =
                TimeProductProverConfig::<F, BenchStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(), s.clone()]),
                    num_variables: bench_args.num_variables,
                    streams: vec![s.clone(), s],
                };
            let transcript = ProductSumcheck::<F>::prove::<
                BenchStream<F>,
                TimeProductProver<F, BenchStream<F>>,
            >(
                &mut TimeProductProver::<F, BenchStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::ProductBlendy => {
            let config: BlendyProductProverConfig<F, BenchStream<F>> =
                BlendyProductProverConfig::<F, BenchStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(), s.clone()]),
                    num_variables: bench_args.num_variables,
                    num_stages: bench_args.stage_size,
                    streams: vec![s.clone(), s],
                };
            let transcript = ProductSumcheck::<F>::prove::<
                BenchStream<F>,
                BlendyProductProver<F, BenchStream<F>>,
            >(
                &mut BlendyProductProver::<F, BenchStream<F>>::new(config),
                &mut rng,
            );
            assert!(transcript.is_accepted);
        }
        AlgorithmLabel::ProductCTY => {
            let config: SpaceProductProverConfig<F, BenchStream<F>> =
                SpaceProductProverConfig::<F, BenchStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(), s.clone()]),
                    num_variables: bench_args.num_variables,
                    streams: vec![s.clone(), s],
                };
            let transcript = ProductSumcheck::<F>::prove::<
                BenchStream<F>,
                SpaceProductProver<F, BenchStream<F>>,
            >(
                &mut SpaceProductProver::<F, BenchStream<F>>::new(config),
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
            // run_on_field::<BN254Field>(bench_args);
            run_on_field::<F64>(bench_args);
        }
    };
}
