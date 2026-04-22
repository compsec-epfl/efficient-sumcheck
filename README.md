<h1 align="center">Efficient Sumcheck</h1>

<p align="center">
  <a href="CHANGELOG.md"><img alt="version" src="https://img.shields.io/badge/version-0.0.2-blue"></a>
  <a href="LICENSE-MIT"><img alt="license" src="https://img.shields.io/badge/license-MIT%2FApache--2.0-green"></a>
</p>

A high-performance sumcheck library with [correctness-fuzzing](#correctness) against a verified oracle.

- **Efficient** — transparent SIMD acceleration (8-wide AVX-512, 2-wide NEON)
- **Streaming-capable** — optional sublinear memory via sequential evaluation
- **Complete** — built-in Fiat-Shamir, partial execution, per-round hooks

Built using [arkworks](https://github.com/arkworks-rs). Compatible with any ecosystem — see [`docs/compatibility.md`](docs/compatibility.md).

## Quick Start

### Multilinear Sumcheck

Proves $H = \sum_{x \in \lbrace 0,1 \rbrace^v} p(x)$ where $p$ is a multilinear polynomial.

```rust
use effsc::{noop_hook, runner::sumcheck};
use effsc::provers::multilinear::MultilinearProver;

let mut prover = MultilinearProver::new(evals);
let proof = sumcheck(
    &mut prover,
    num_vars,
    &mut transcript,
    noop_hook,
);
```

### Inner Product Sumcheck

Proves $H = \sum_{x \in \lbrace 0,1 \rbrace^v} f(x) \cdot g(x)$ for two multilinear polynomials. Degree-2 round polynomials.

```rust
use effsc::{noop_hook, runner::sumcheck};
use effsc::provers::inner_product::InnerProductProver;

let mut prover = InnerProductProver::new(a, b);
let proof = sumcheck(
    &mut prover,
    num_vars,
    &mut transcript,
    noop_hook,
);
```

### Coefficient Sumcheck

Proves $H = \sum_{x \in \lbrace 0,1 \rbrace^v} p(x)$ where $\deg_{x_i}(p) \leq d$. The user implements `RoundPolyEvaluator` to define per-pair round polynomial contributions; the library handles iteration, parallelism, and reductions.

```rust
use effsc::{noop_hook, runner::sumcheck};
use effsc::provers::coefficient::CoefficientProver;

let mut prover = CoefficientProver::new(
    &evaluator,
    tablewise,
    pairwise,
);
let proof = sumcheck(
    &mut prover,
    num_rounds,
    &mut transcript,
    noop_hook,
);
```

### Eq-Factored Sumcheck

Proves $H = \sum_{x \in \lbrace 0,1 \rbrace^v} \mathrm{eq}(w, x) \cdot p(x)$ for a fixed point $w \in F^v$ and a multilinear polynomial $p$. Degree-2 round polynomials. Shows up in lookup arguments and any reduction that couples a public point to a witness polynomial via the multilinear equality predicate.

```rust
use effsc::{noop_hook, runner::sumcheck};
use effsc::provers::eq_factored::EqFactoredProver;

let mut prover = EqFactoredProver::new(w, p_evals);
let proof = sumcheck(
    &mut prover,
    num_vars,
    &mut transcript,
    noop_hook,
);
// final_value = p(r) · eq(w, r)
let (p_r, eq_wr) = prover.final_factors();
```

### Verification

One verifier for any degree $d$. Returns `SumcheckResult { challenges, final_claim }` — ⚠️ the caller is responsible for the oracle check ([Thaler Remark 4.2](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)).

```rust
use effsc::{noop_hook_verify, verifier::sumcheck_verify};

let result = sumcheck_verify(
    claimed_sum,
    degree,
    num_rounds,
    &mut transcript,
    noop_hook_verify,
)?;

// Standalone: compare against the prover's claimed final value.
assert_eq!(result.final_claim, proof.final_value);

// Composed (WHIR, GKR): pass final_claim to the next layer.
next_layer_claim = result.final_claim;
```

## Variable Ordering

Each prover comes in two variants:

- **MSB** (half-split) — optimal memory layout in most cases. Used by WHIR and WARP.
- **LSB** (pair-split) — optimal for streaming applications where evaluations arrive sequentially.

| MSB | LSB |
|-----|-----|
| `MultilinearProver` | `MultilinearProverLSB` |
| `InnerProductProver` | `InnerProductProverLSB` |
| `CoefficientProver` | `CoefficientProverLSB` |
| `EqFactoredProver` | — |
| `GkrProver` | — |

See [`docs/design.md`](docs/design.md) for details.

## Partial Execution and Hooks

The `sumcheck()` runner supports partial execution (`num_rounds < v`) and per-round hooks for composed protocols:

```rust
// WHIR: partial sumcheck with proof-of-work grinding
let proof = sumcheck(
    &mut prover,
    folding_factor,  // num_rounds < v
    &mut transcript,
    |_, t| round_pow.prove(t),  // per-round hook
);
```

## SIMD Acceleration

All provers transparently auto-dispatch to SIMD backends. Supported fields:

- [x] Goldilocks ($p = 2^{64} - 2^{32} + 1$) and degree-2/3 extensions
- [ ] M31 ($p = 2^{31} - 1$) and extensions
- [ ] BabyBear ($p = 2^{31} - 2^{27} + 1$) and extensions
- [ ] KoalaBear ($p = 2^{31} - 2^{24} + 1$) and extensions

| Backend | Width | Platform |
|---------|-------|----------|
| NEON | 2-wide | aarch64 (Apple M-series, Graviton) |
| AVX-512 IFMA | 8-wide | x86_64 (Sapphire Rapids) |

Scalar for other fields. See [`SECURITY.md`](SECURITY.md#unsafe-code).

## Examples
Integrated into Whir ([PR](https://github.com/WizardOfMenlo/whir/pull/250)) and Warp ([PR](https://github.com/compsec-epfl/warp/pull/24)) with measured performance improvements. Integration capability for streaming contexts like [Jolt](https://github.com/a16z/jolt) is described in [`docs/design.md`](docs/design.md).

## Correctness

🚧 Undergoing fuzzing over randomized inputs against [z-tech/sumcheck-lean4](https://github.com/z-tech/sumcheck-lean4), an oracle with machine-checked proofs of completeness and soundness. Findings to follow.

## References

[[LFKN92](https://dl.acm.org/doi/pdf/10.1145/146585.146605)]: Carsten Lund, Lance Fortnow, Howard J. Karloff, and Noam Nisan. "Algebraic Methods for Interactive Proof Systems". In: Journal of the ACM 39.4 (1992).

[[CTY11](https://arxiv.org/pdf/1109.6882.pdf)]: Graham Cormode, Justin Thaler, and Ke Yi. "Verifying computations with streaming interactive proofs". In: Proceedings of the VLDB Endowment 5.1 (2011), pp. 25-36.

[[VSBW13](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6547112)]: Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. "A hybrid architecture for interactive verifiable computation". In: Proceedings of the 34th IEEE Symposium on Security and Privacy. Oakland '13. 2013, pp. 223-237.

[[CFFZ24](https://eprint.iacr.org/2024/524.pdf)]: Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Andrew Zitek-Estrada. "A time-space tradeoff for the sumcheck prover". In: Cryptology ePrint Archive.

[[BCFFMMZ25](https://eprint.iacr.org/2025/1473.pdf)]: Anubhav Baweja, Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Pratyush Mishra, Tushar Mopuri, and Andrew Zitek-Estrada. "Time-Space Trade-Offs for Sumcheck". In: TCC Theory of Cryptography: 23rd International Conference, pp. 37.

[[Thaler23](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)]: Justin Thaler. "Proofs, Arguments, and Zero-Knowledge". Chapter 4: Interactive Proofs. July 2023.

[[BDDT25](https://eprint.iacr.org/2025/1117.pdf)]: Aarushi Bagad, Quang Dao, Yuri Domb, and Justin Thaler. "Speeding Up Sum-Check Proving". Cryptology ePrint Archive, 2025/1117.
