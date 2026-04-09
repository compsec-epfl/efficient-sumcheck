# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **SIMD auto-dispatch** for Goldilocks (NEON + AVX-512 IFMA) across all three sumcheck variants.
- **`poly_ops` module** — zero-allocation polynomial arithmetic on coefficient slices.
- **`RoundPolyEvaluator` trait** for `coefficient_sumcheck` — user implements per-pair math, library handles iteration, parallelism, and reductions.
- **Base/Extension field support** (`<BF, EF>`) for `multilinear_sumcheck` and `inner_product_sumcheck`.

### Changed
- **Inner product sumcheck**: 2 prover messages per round instead of 3 (verifier derives the third).
- **Coefficient sumcheck**: sends d coefficients per round instead of d+1.
- **`protogalaxy::fold`**: rewritten with flat buffers (93× faster at scale).
- **`coefficient_sumcheck`** takes `&impl RoundPolyEvaluator<F>` instead of a closure.

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
