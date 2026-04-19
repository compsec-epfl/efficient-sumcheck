# Security

## Audit status

This library has **not undergone a formal security audit**. It is research-grade
software under active development.

## Threat model

Sumcheck is public-coin: prover messages depend only on the polynomial and
verifier challenges. This library implements plain (non-ZK) sumcheck.

- **Standalone use** (`g` public): the transcript reveals everything the
  prover computes on. Timing side channels leak nothing beyond the transcript.

- **ZK embedding** (`g` encodes witness data; ZK supplied by surrounding
  commitments/masking): the prover's arithmetic runs on secrets. A
  transcript-observing adversary is still safe, but **resistance to local
  side-channel adversaries has not been formally verified**. Fixed-size
  Montgomery multiplication is inherently data-independent, but this has
  not been audited, and no constant-time claim is made for arbitrary
  `SumcheckField` implementations. Callers in that threat model must
  supply constant-time field operations.

## Oracle check responsibility

`sumcheck_verify` checks round consistency and returns
`SumcheckResult { challenges, final_claim }`. It does **not** verify
that `final_claim == g(r_1, ..., r_v)` — this oracle check ([Thaler Remark 4.2](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)) is the caller's responsibility.

**Forgetting the oracle check is a soundness bug.** A malicious prover
can craft round polynomials that pass all consistency checks but reduce
to an arbitrary final claim. Without the oracle check, the verifier
accepts.

Correct usage depends on the protocol context:

| Context | What the caller must do |
|---------|------------------------|
| Standalone | `assert_eq!(result.final_claim, proof.final_value)` |
| Composed (WHIR, GKR) | Pass `result.final_claim` to the next layer, which checks it |
| Custom (WARP) | Compute expected value from `result.challenges` and compare |

## `unsafe` code

Outside of the SIMD path, the library contains **no `unsafe` code**. Within the
SIMD subsystem, `unsafe` is confined to two categories:

1. **SIMD intrinsics** (`core::arch`) — `_mm512_loadu_si512`, `vld1q_u64`, etc.
   These are `unsafe` by definition in Rust; there is no safe alternative. They
   appear exclusively in the backend kernels (`avx512.rs`, `neon.rs`) and the
   evaluate/reduce loops that call them.

2. **Field ↔ `u64` reinterpretation** — arkworks field types don't derive
   `zerocopy`, so the blanket `SumcheckField` impl for `ark_ff::Field` uses
   `transmute_copy` and `from_raw_parts` to reinterpret Goldilocks elements as
   their underlying Montgomery-form `u64` values. These are centralized in five
   trait methods (`_to_raw_u64`, `_from_raw_u64`, `_as_u64_slice`,
   `_as_u64_slice_mut`, `_from_u64_components`) in `field.rs`, each with a
   SAFETY comment documenting the invariant. The dispatch layer itself contains
   no `unsafe`.

Non-arkworks types using `SimdRepr` avoid category 2 entirely — the `zerocopy`
bounds (`IntoBytes + FromBytes + Immutable`) provide compile-time layout
verification, so no `unsafe` reinterpretation is needed.

The scalar (non-SIMD) code path uses no `unsafe` at all. If `F` is not a
recognised Goldilocks field, SIMD dispatch is skipped and the entire protocol
runs in safe Rust.

## Reporting a vulnerability

If you discover a security issue, please report it responsibly via
[private vulnerability reporting](https://github.com/compsec-epfl/space-efficient-sumcheck/security/advisories/new)
on this repository.

**Do not** open a public GitHub issue for security vulnerabilities.
