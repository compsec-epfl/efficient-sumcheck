# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Base/Extension field support**: `multilinear_sumcheck` and `inner_product_sumcheck` now take two type parameters `<BF, EF>` — base field for evaluations, extension field for challenges. Set `EF = BF` when no extension is needed.
- `pairwise::cross_field_reduce` — parallel helper for folding `BF` evaluations with an `EF` challenge.

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
