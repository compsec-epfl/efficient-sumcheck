# Migrating from arkworks-rs/sumcheck

If you're using [arkworks-rs/sumcheck](https://github.com/arkworks-rs/sumcheck),
here's how to accomplish the same things with effsc.

## If you're using `ListOfProductsOfPolynomials`

In arkworks, you describe a sumcheck claim as a sum of products of multilinear
extensions and let the library evaluate it generically. In effsc, you implement
`RoundPolyEvaluator` — writing the polynomial logic directly. The library
handles folding, parallelism, and SIMD.

For example, given $H = \sum_{x} [ f(x) \cdot g(x) + h(x) \cdot k(x) ]$:

```rust
use effsc::coefficient_sumcheck::RoundPolyEvaluator;

struct MyEvaluator;

impl RoundPolyEvaluator<F> for MyEvaluator {
    fn degree(&self) -> usize { 2 }

    fn accumulate_pair(
        &self,
        coeffs: &mut [F],
        _tw: &[(&[F], &[F])],
        pw: &[(F, F)],
    ) {
        let (f0, f1) = pw[0];
        let (g0, g1) = pw[1];
        let (h0, h1) = pw[2];
        let (k0, k1) = pw[3];

        coeffs[0] += f0 * g0 + h0 * k0;
        let at_1 = f1 * g1 + h1 * k1;
        coeffs[1] += at_1 - coeffs[0];
        let f2 = f0 + (f1 - f0) + (f1 - f0);
        let g2 = g0 + (g1 - g0) + (g1 - g0);
        let h2 = h0 + (h1 - h0) + (h1 - h0);
        let k2 = k0 + (k1 - k0) + (k1 - k0);
        coeffs[2] += f2 * g2 + h2 * k2;
    }
}
```

Then wire it through `CoefficientProver`:

```rust
use effsc::provers::coefficient::CoefficientProver;

let mut prover = CoefficientProver::new(
    &MyEvaluator,
    &[],
    &mut [f_evals, g_evals, h_evals, k_evals],
);
```

The evaluator reads as a direct translation of the protocol math, and because
you write the polynomial logic yourself, you can exploit structure like shared
subexpressions or known-constant factors.

## If you're using `GKRRoundSumcheck`

In arkworks, `GKRRoundSumcheck` wraps `MLSumcheck` for the GKR round
polynomial. In effsc, use `GkrProver`:

```rust
use effsc::provers::gkr::GkrProver;

// add_evals, mult_evals: gate predicates over {0,1}^{2k},
//   partially evaluated at the previous layer's random point.
// w_evals: witness W_{i+1} over {0,1}^k.
let mut prover = GkrProver::new(add_evals, mult_evals, w_evals);
let proof = sumcheck(
    &mut prover,
    2 * k,
    &mut transcript,
    noop_hook,
);

// Extract claimed W values for the reduce-to-one sub-protocol.
let (w_b_star, w_c_star) = prover.claimed_w_values();
```

A full GKR verification loops one sumcheck per circuit layer, feeding each
layer's `(w_b_star, w_c_star)` into the reduce-to-one sub-protocol and then
into the next layer's sumcheck.

The current `GkrProver` has the same O(2^{2k}) complexity per layer as
arkworks-rs/sumcheck. An optimized O(2^k · k) version using incremental
eq-polynomial bookkeeping is possible but not yet implemented.
