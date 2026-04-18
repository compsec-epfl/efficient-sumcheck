<h1 align="center">Efficient Sumcheck</h1>

A high-performance sumcheck library with transparent SIMD acceleration, built-in Fiat-Shamir support, streaming capability, and correctness fuzzed against an [oracle](https://github.com/z-tech/sumcheck-lean4) with formalized completeness and soundness.

This library was built for [arkworks](https://github.com/arkworks-rs) and compatible with any ecosystem that implements the [`SumcheckField`](#generic-field) trait.

**Security:** This library has not undergone a security audit.

## Quick Start

### Multilinear Sumcheck

Proves $H = \displaystyle\sum_{x \in \lbrace 0,1 \rbrace^v} p(x)$ where $p$ is a multilinear polynomial.

```rust
use effsc::{no_hook, provers::multilinear::MultilinearProver, runner::sumcheck, transcript::TestTranscript};

let evals: Vec<F> = /* */;
let num_vars = evals.len().trailing_zeros() as usize;

let mut prover = MultilinearProver::new(evals);
let mut transcript = TestTranscript::new(&mut rng);
let proof = sumcheck(&mut prover, num_vars, &mut transcript, no_hook);

// proof.round_polys: Vec<Vec<F>> -- g_j at {0, 1, ..., deg}
// proof.challenges: Vec<F>       -- r_1, ..., r_v
// proof.final_value: F           -- g(r_1, ..., r_v)
```

### Inner Product Sumcheck

Proves $H = \displaystyle\sum_{x \in \lbrace 0,1 \rbrace^v} f(x) \cdot g(x)$ for two multilinear polynomials. Degree-2 round polynomials.

```rust
use effsc::{provers::inner_product::InnerProductProver, runner::sumcheck};

let mut prover = InnerProductProver::new(a, b);
let proof = sumcheck(&mut prover, num_vars, &mut transcript, no_hook);

// Prover-specific post-state:
let (f_at_r, g_at_r) = prover.final_evaluations();
```

### Coefficient Sumcheck

Proves $H = \displaystyle\sum_{x \in \lbrace 0,1 \rbrace^v} p(x)$ where $\deg_{x_i}(p) \leq d$. The user implements `RoundPolyEvaluator` to define per-pair round polynomial contributions; the library handles iteration, parallelism, and reductions.

```rust
use effsc::coefficient_sumcheck::{coefficient_sumcheck, RoundPolyEvaluator};

struct MyEvaluator;
impl RoundPolyEvaluator<F> for MyEvaluator {
    fn degree(&self) -> usize { 1 }
    fn accumulate_pair(
        &self,
        coeffs: &mut [F],
        tw: &[(&[F], &[F])],
        pw: &[(F, F)],
    ) {
        let (even, odd) = pw[0];
        coeffs[0] += even;
        coeffs[1] += odd - even;
    }
}

let result = coefficient_sumcheck(
    &MyEvaluator,
    &mut tablewise,
    &mut pairwise,
    num_rounds,
    &mut transcript,
);
```

Or via the `SumcheckProver` trait:

```rust
use effsc::{provers::coefficient::CoefficientProver, runner::sumcheck};

let mut prover = CoefficientProver::new(&MyEvaluator, tablewise, pairwise);
let proof = sumcheck(&mut prover, num_rounds, &mut transcript, no_hook);
```

### Verification

One verifier for any degree $d$:

```rust
use effsc::verifier::sumcheck_verify;

let result = sumcheck_verify(claimed_sum, degree, num_rounds, &mut transcript, no_hook_verify);
match result {
    Ok((final_claim, challenges)) => { /* caller checks final_claim = g(r) */ }
    Err(e) => { /* consistency check failed */ }
}
```

### Fold

The fold primitive is exposed for callers that need it independently of the sumcheck protocol (e.g., WHIR's `multilinear_fold`):

$$\tilde{p}(x_2, \ldots, x_v) = r_1 \cdot \tilde{p}(1, x_2, \ldots, x_v) + (1 - r_1) \cdot \tilde{p}(0, x_2, \ldots, x_v)$$

```rust
use effsc::fold;

fold(&mut evals, challenge);  // MSB half-split, SIMD-accelerated
```

## The Prover Trait

All provers implement `SumcheckProver<F>`:

```rust
pub trait SumcheckProver<F: SumcheckField> {
    fn degree(&self) -> usize;
    fn round(&mut self, challenge: Option<F>) -> Vec<F>;
    fn finalize(&mut self, last_challenge: F);
    fn final_value(&self) -> F;
}
```

The protocol runner calls `round()` once per round. The caller retains `&mut P` ownership after sumcheck and can query prover-specific post-state (e.g., folded vectors for WHIR, claimed $W$ values for GKR).

## Generic Field

The library is generic over any type implementing `SumcheckField`:

```rust
pub trait SumcheckField:
    Copy + Send + Sync + PartialEq + Debug
    + Add + Sub + Mul + Neg + AddAssign + SubAssign + MulAssign
    + Sum + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn from_u64(val: u64) -> Self;
    fn inverse(&self) -> Option<Self>;
}
```

A blanket implementation for all `ark_ff::Field` types is provided when the `arkworks` feature is enabled (default). Non-arkworks users compile with `--no-default-features` and implement the trait for their own field type.

## Variable Ordering

Two layouts are supported, corresponding to different data availability patterns:

| Prover | Layout | Fold pairing | Best for |
|--------|--------|-------------|----------|
| `MultilinearProver` | MSB (half-split) | $(v_k,\ v_{k+L/2})$ | In-memory, WHIR |
| `MultilinearProverLSB` | LSB (pair-split) | $(v_{2k},\ v_{2k+1})$ | Sequential streaming, Jolt |
| `InnerProductProver` | MSB | $(a_k \cdot b_k,\ a_{k+L/2} \cdot b_{k+L/2})$ | In-memory, WHIR |
| `InnerProductProverLSB` | LSB | $(a_{2k} \cdot b_{2k},\ a_{2k+1} \cdot b_{2k+1})$ | Sequential streaming, Jolt |
| `CoefficientProver` | MSB | half-split pairs | In-memory, WARP |
| `CoefficientProverLSB` | LSB | adjacent pairs | Sequential streaming, Jolt |

MSB is optimal when the full table is in memory or seekable on disk. LSB is optimal when evaluations arrive incrementally (e.g., Jolt CPU trace) — adjacent pairs are immediately available for folding.

## Partial Execution and Hooks

The `sumcheck()` runner supports partial execution (`num_rounds < v`) and per-round hooks for composed protocols:

```rust
// WHIR: partial sumcheck with proof-of-work grinding
let proof = sumcheck(
    &mut InnerProductProver::new(a, b),
    folding_factor,
    &mut transcript,
    |_, t| round_pow.prove(t),
);
```

GKR compatibility (custom `SumcheckProver` impls with post-state inspection) is planned — see [`docs/design.md` section 10](docs/design.md).

## SIMD Acceleration

All provers auto-dispatch to SIMD-accelerated backends. Supported fields:

- [x] Goldilocks ($p = 2^{64} - 2^{32} + 1$) and degree-2/3 extensions
- [ ] M31 ($p = 2^{31} - 1$) and extensions
- [ ] BabyBear ($p = 2^{31} - 2^{27} + 1$) and extensions
- [ ] KoalaBear ($p = 2^{31} - 2^{24} + 1$) and extensions

| Backend | Width | Platform |
|---------|-------|----------|
| NEON | 2-wide | aarch64 (Apple M-series, Graviton) |
| AVX-512 IFMA | 8-wide | x86_64 (Sapphire Rapids) |

Detection is constant-folded by LLVM after monomorphization. Zero overhead on non-Goldilocks fields.

Non-arkworks Goldilocks types opt into SIMD via the `SimdRepr` trait, whose layout safety is compiler-verified by `zerocopy`:

```rust
#[derive(zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
#[repr(transparent)]
struct MyGoldilocks(u64);

impl SimdRepr for MyGoldilocks {
    fn modulus() -> u64 { GOLDILOCKS_P }
}
```

To enable AVX-512:
```bash
RUSTFLAGS="-C target-feature=+avx512ifma" cargo build --release
```

## Zero-Allocation Polynomial Arithmetic

> **Note:** `poly_ops` will be upstreamed to arkworks soon.

The `poly_ops` module provides slice-based polynomial arithmetic:

```rust
use effsc::poly_ops;

let mut out = [F::ZERO; 3];
poly_ops::mul_into(&mut out, &a, &b);
poly_ops::add_scaled(&mut out, scalar, &c);
let val = poly_ops::eval_at(&out, challenge);
```

## Features

```toml
[features]
default = ["arkworks", "parallel"]
arkworks = ["ark-ff", "ark-poly", "ark-serialize", "ark-std", "spongefish"]
parallel = ["rayon"]
```

- `arkworks` (default): blanket `SumcheckField` impl for `ark_ff::Field`
- `parallel` (default): rayon parallelism for fold and round computation
- `--no-default-features`: pure `SumcheckField` library, no arkworks dependency

## Benchmarks

```bash
cargo bench --bench sumcheck
```

Benchmark matrix: `{multilinear, inner_product} x {F64, F64Ext3} x {2^16, 2^20, 2^24}` plus fold kernel throughput. CI tracks regressions via [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark).

## Integration Examples

### WHIR

[WHIR](https://github.com/WizardOfMenlo/whir) uses partial inner-product sumcheck with per-round proof-of-work hooks. See [`docs/design.md` section 11](docs/design.md).

### WARP

[WARP](https://github.com/compsec-epfl/warp) uses `CoefficientProver` with `folding::protogalaxy::fold` to batch codeword + R1CS constraint checks into a single sumcheck.

### Jolt

[Jolt](https://github.com/a16z/jolt) compatibility is via an adapter struct that bridges `SumcheckProver` to Jolt's `SumcheckInstanceProver`. Use `MultilinearProverLSB` / `InnerProductProverLSB` for Jolt's default LSB binding order. See [`docs/design.md` section 12](docs/design.md).

## Prover Strategies

| Strategy | Reference |
|----------|-----------|
| Time (linear time, linear space) | [VSBW13] |
| Blendy (linear time, sublinear space) | [CFFZ24], [BCFFMMZ25] |
| Space (quasilinear time, logarithmic space) | [CTY11] |

## References

[[LFKN92](https://dl.acm.org/doi/pdf/10.1145/146585.146605)]: Carsten Lund, Lance Fortnow, Howard J. Karloff, and Noam Nisan. "Algebraic Methods for Interactive Proof Systems". In: Journal of the ACM 39.4 (1992).

[[CTY11](https://arxiv.org/pdf/1109.6882.pdf)]: Graham Cormode, Justin Thaler, and Ke Yi. "Verifying computations with streaming interactive proofs". In: Proceedings of the VLDB Endowment 5.1 (2011), pp. 25-36.

[[VSBW13](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6547112)]: Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. "A hybrid architecture for interactive verifiable computation". In: Proceedings of the 34th IEEE Symposium on Security and Privacy. Oakland '13. 2013, pp. 223-237.

[[CFFZ24](https://eprint.iacr.org/2024/524.pdf)]: Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Andrew Zitek-Estrada. "A time-space tradeoff for the sumcheck prover". In: Cryptology ePrint Archive.

[[BCFFMMZ25](https://eprint.iacr.org/2025/1473.pdf)]: Anubhav Baweja, Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Pratyush Mishra, Tushar Mopuri, and Andrew Zitek-Estrada. "Time-Space Trade-Offs for Sumcheck". In: TCC Theory of Cryptography: 23rd International Conference, pp. 37.

[[Thaler23](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)]: Justin Thaler. "Proofs, Arguments, and Zero-Knowledge". Chapter 4: Interactive Proofs. July 2023.

[[BDDT25](https://eprint.iacr.org/2025/1117.pdf)]: Aarushi Bagad, Quang Dao, Yuri Domb, and Justin Thaler. "Speeding Up Sum-Check Proving". Cryptology ePrint Archive, 2025/1117.
