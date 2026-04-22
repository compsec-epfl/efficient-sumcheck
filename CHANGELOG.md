# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — Canonical Rewrite

Major revision: unified API, one verifier, one proof type, 7 provers, SIMD acceleration.

### Breaking

- **Package renamed** from `efficient-sumcheck` to `effsc`.
- **Single verifier** — `sumcheck_verify()` returns `SumcheckResult { challenges, final_claim }`. The oracle check is the caller's responsibility ([Thaler Remark 4.2](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)). Removed `inner_product_sumcheck_verify`, `multilinear_sumcheck_verify`, and `coefficient_sumcheck::sumcheck_verify`.
- **Single proof type** — `SumcheckProof<F>` replaces `Sumcheck<F>` and `ProductSumcheck<F>`.
- **Transcript redesigned** — `send()`/`receive()`/`challenge()` replace `read()`/`write()`.
- **Legacy entry points demoted** — use `runner::sumcheck()` with a prover type.
- **Wire format: EvalsInfty.** `d` values per round instead of `d + 1`; consistency is now structural. Details in [docs/design.md §7a](docs/design.md).

### Added

- **`SumcheckProver<F>` trait** — single extension point for all polynomial shapes.
- **7 concrete provers** — `MultilinearProver`, `InnerProductProver`, `CoefficientProver` (each with MSB + LSB variants), `GkrProver`.
- **`SumcheckField` trait** — generic field interface; blanket impl for `ark_ff::Field` behind `feature = "arkworks"`.
- **`SimdRepr` trait** — safe SIMD opt-in with `zerocopy` layout verification.
- **`runner::sumcheck()`** — single runner with partial execution and per-round hooks.
- **Eq polynomial utilities** — `eq_poly`, `eq_poly_non_binary`, O(2^v) incremental `compute_hypercube_eq_evals`.
- **Adversarial verifier tests** — corrupted proofs, wrong sums, wrong final values across all prover types.
- **`no_std` support** — core library works without `arkworks` feature.
- **SIMD** — transparent 8-wide AVX-512 IFMA, 2-wide NEON acceleration.

### Integrations

- [WHIR](https://github.com/WizardOfMenlo/whir) ([PR #250](https://github.com/WizardOfMenlo/whir/pull/250))
- [WARP](https://github.com/compsec-epfl/warp) ([PR #24](https://github.com/compsec-epfl/warp/pull/24))

### Removed

- **~4,500 lines of legacy code** — old `Prover` trait, `TimeProver`/`SpaceProver`/`BlendyProver`, `OrderStrategy`, `messages/`, `interpolation/`, `simd_ops`.

## [0.0.2] - 2026-02-11

### Added
- **High-level API**: `multilinear_sumcheck()` and `inner_product_sumcheck()` free functions for simple one-call sumcheck.
- **Fiat–Shamir support**: `Transcript` trait with `SpongefishTranscript` (real Fiat-Shamir via [SpongeFish](https://github.com/arkworks-rs/spongefish)) and `SanityTranscript` (random challenges for testing).
- `batched_constraint_poly` — merges dense and sparse polynomials into a single constraint polynomial.
- **Pairwise reduce for `TimeProductProver`** ([#87](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/87)).
- **Improved order strategies** ([#86](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/86)).
- **Pairwise compression for `TimeProver`** ([#83](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/83)).
- `BasicProver` for testing ([#84](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/84)).

### Changed
- Removed `claim` from the `Prover` trait ([#85](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/85)).
- Updated to criterion 0.8 ([#88](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/88)).

## [0.0.1] - 2025-08-26

### Added
- **Rayon parallelization** for `TimeProver` and `TimeProductProver` ([#80](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/80)).
- **Stream iterator** for sequential evaluation access ([#74](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/74)).
- **Benchmark improvements** ([#75](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/75), [#76](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/76)).
- `FileStream` for memory-mapped file-backed evaluations ([#67](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/67)).
- **Blendy product prover** — `BlendyProductProver` ([#64](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/64)).
- Refactored module structure ([#63](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/63)).

### Changed
- Bumped arkworks dependencies from 0.4 → 0.5.
- Avoid dynamic dispatch (`dyn`) and heap allocation (`Box`) ([#53](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/53)).
- Gray code ordering for sequential Lagrange polynomial iterator ([#55](https://github.com/compsec-epfl/space-efficient-sumcheck/pull/55)).

## [0.0.0] - 2024-02-09

### Added
- Initial release with three proving algorithms:
  - **`SpaceProver`** — quasi-linear time, logarithmic space [[CTY11](https://arxiv.org/pdf/1109.6882.pdf)].
  - **`TimeProver`** — linear time, linear space [[VSBW13](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6547112)].
  - **`BlendyProver`** — linear time, sublinear space [[CFFZ24](https://eprint.iacr.org/2024/524.pdf)].
- Product sumcheck variants (`SpaceProductProver`, `TimeProductProver`, `BlendyProductProver`).
- `MemoryStream` for in-memory evaluation access.
- Hypercube utilities and Lagrange polynomial interpolation.
- Configurable reduction modes (pairwise, variablewise).
- Benchmark suite using criterion.
