# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — Canonical Rewrite

### Breaking

- **Package renamed** from `efficient-sumcheck` to `effsc`.
- **`Transcript` trait redesigned** — `read()`/`write()` replaced with `send()`/`receive()`/`challenge()` to distinguish prover messages from verifier challenges. Enables correct spongefish `VerifierState` interop.
- **Legacy entry points demoted** — `multilinear_sumcheck()`, `inner_product_sumcheck()` are now `pub(crate)`. Use `runner::sumcheck()` with a prover type instead.
- **`Sumcheck<F>` and `ProductSumcheck<F>`** replaced by unified `SumcheckProof<F>`.
- **Order strategies removed** — MSB-only for in-memory workloads; LSB variants added for streaming.

### Added

- **`SumcheckProver<F>` trait** — single extension point for all prover strategies and polynomial shapes. Methods: `degree()`, `round()`, `finalize()`, `final_value()`.
- **`SumcheckField` trait** — generic field interface (not arkworks-specific). Blanket impl for `ark_ff::Field` behind `feature = "arkworks"` (default).
- **`SimdRepr` trait** — safe SIMD opt-in for non-arkworks field types, with layout safety compiler-verified by `zerocopy`.
- **`runner::sumcheck()`** — single protocol runner for any `SumcheckProver`, with partial execution and per-round hooks.
- **`verifier::sumcheck_verify()`** — degree-generic verifier with Lagrange interpolation and `Result`-based error handling.
- **`SumcheckProof<F>`** — unified proof type replacing `Sumcheck<F>` and `ProductSumcheck<F>`.
- **`SumcheckError`** — structured error type with `TranscriptError` variant.
- **`noop_hook`** — named no-op hook function (replaces `|_, _| {}`).
- **`TestTranscript`** — renamed from `SanityTranscript` (old name kept as alias).
- **6 concrete provers** (3 shapes × 2 orderings):
  - `MultilinearProver` / `MultilinearProverLSB` (degree 1)
  - `InnerProductProver` / `InnerProductProverLSB` (degree 2)
  - `CoefficientProver` / `CoefficientProverLSB` (degree d)
- **MSB + LSB variable ordering** — MSB (half-split) for in-memory/WHIR, LSB (pair-split) for sequential streaming/Jolt.
- **`arkworks` feature gate** — all ark dependencies optional; non-arkworks users compile with `--no-default-features`.
- **`zerocopy` dependency** — for `SimdRepr` layout verification.
- **CI workflows** — `ci.yml` (build + clippy + test), `bench.yml` (criterion regression tracking with auto SIMD detection).
- **Criterion benchmark harness** — `{multilinear, inner_product} × {F64, F64Ext3} × {2^16, 2^20, 2^24}` + fold throughput.
- **Design document** (`docs/design.md`) — 17-section specification based on Thaler Chapter 4.
- **Slide deck** (`docs/slides.md`, `docs/slides.pdf`) — 30-slide presentation.

### Changed

- **SIMD dispatch** uses `SumcheckField::_simd_field_config()` instead of `ark_ff::Field::BasePrimeField::MODULUS`. Same constant-folding behavior, no arkworks dependency in the dispatch path.
- **`CoefficientProver`** (MSB) uses half-split pairing; `CoefficientProverLSB` uses adjacent pairing. Both implement `SumcheckProver<F>`.
- **`reorder_vec`** simplified to `reorder_vec_msb` (bit-reversal only, no order strategy parameter).

### Removed

- **~4,500 lines of legacy code**: old `Prover` trait, `TimeProver`/`SpaceProver`/`BlendyProver` (multilinear + product), `OrderStrategy` (ascending/descending/graycode), `messages/`, `interpolation/`, `Hypercube`/`HypercubeMember`, `StreamIterator`.
- **`simd_ops` module** — functionality merged into provers and fold.
- **Old benchmark files** (`provers.rs`, `simd_vs_generic.rs`).

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
