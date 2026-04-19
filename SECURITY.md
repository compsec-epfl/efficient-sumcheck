# Security Policy

## Audit status

This library has **not undergone a formal security audit**. It is research-grade
software under active development.

## Threat model

The sumcheck protocol is **public-coin**: the prover's computation depends only
on public polynomial evaluations and verifier challenges. No secret values flow
through the prover's arithmetic, so timing side channels in the field operations
do not leak private information.

If zero-knowledge sumcheck (blinded/masked variants) is added in the future, a
timing analysis of the field arithmetic layer would be warranted. The
fixed-size Montgomery multiplication used for Goldilocks is inherently
data-independent, but this property has not been formally verified.

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

If you discover a security issue, please report it responsibly via one of:

- **GitHub:** Use [private vulnerability reporting](https://github.com/compsec-epfl/space-efficient-sumcheck/security/advisories/new) on this repository
- **Email:** andrew.zitek@epfl.ch (subject: `[effsc] Security vulnerability report`)

Please include a description of the issue, its potential impact, and steps to
reproduce if applicable. You will receive an acknowledgement within 72 hours.

**Do not** open a public GitHub issue for security vulnerabilities.
