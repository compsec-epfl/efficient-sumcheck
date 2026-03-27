//! SIMD auto-dispatch for the multilinear sumcheck protocol.
//!
//! When `BF == EF == Goldilocks F64`, the sumcheck is transparently routed
//! to a double-buffered Montgomery-arithmetic backend using NEON intrinsics.
//!
//! The TypeId checks evaluate to compile-time constants in monomorphized code,
//! so LLVM eliminates the dead branch — zero cost for non-matching types.

use ark_ff::Field;

use crate::multilinear::Sumcheck;
use crate::transcript::Transcript;

#[cfg(target_arch = "aarch64")]
use crate::simd_fields::goldilocks::mont_neon::MontGoldilocksNeon as MontBackend;
#[cfg(target_arch = "aarch64")]
use crate::simd_fields::SimdBaseField;

/// Returns `true` when `T` is a Goldilocks field type (q = 2^64 - 2^32 + 1)
/// stored as a single Montgomery-form `u64`.
///
/// Matches **both** representations:
/// - `SmallFp<F64Config>` (via `define_field!`) — bare `u64` + PhantomData
/// - `Fp64<MontBackend<FpF64Config, 1>>` (via `MontConfig`) — `BigInt<1>` + PhantomData
///
/// Both have identical memory layout (one `u64` in Montgomery form),
/// so the SIMD arithmetic works for either.
///
/// This is a compile-time constant after monomorphization — LLVM
/// eliminates the dead branch entirely (zero runtime cost).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn is_goldilocks_f64<T: 'static>() -> bool {
    use crate::tests::FpF64;
    use crate::tests::F64; // SmallFp<F64Config> // Fp64<MontBackend<FpF64Config, 1>>
    let tid = std::any::TypeId::of::<T>();
    tid == std::any::TypeId::of::<F64>() || tid == std::any::TypeId::of::<FpF64>()
}

// ─── Auto-dispatch ──────────────────────────────────────────────────────────

/// Try to dispatch to the SIMD backend when `BF == EF` and `BF` is a known
/// SIMD-accelerated type (currently: Goldilocks F64).
///
/// Returns `Some(result)` if the SIMD path was taken, `None` otherwise.
///
/// Zero allocation: transmutes `&mut [BF]` → `&mut [u64]` in-place.
/// The TypeId checks are compile-time constants in monomorphized code.
#[cfg(target_arch = "aarch64")]
pub(crate) fn try_simd_dispatch<BF: Field, EF: Field + From<BF>>(
    evaluations: &mut [BF],
    transcript: &mut impl Transcript<EF>,
) -> Option<Sumcheck<EF>> {
    if !(is_goldilocks_f64::<BF>() && is_goldilocks_f64::<EF>()) {
        return None;
    }

    // BF == EF == F64 (verified via TypeId).

    // SAFETY: F64 is repr-transparent over u64 (Montgomery form).
    // Zero-copy transmute — work directly on the caller's buffer.
    let buf: &mut [u64] = unsafe {
        core::slice::from_raw_parts_mut(evaluations.as_mut_ptr() as *mut u64, evaluations.len())
    };

    // Single closure for transcript round-step: write (s0, s1), return challenge.
    let result_f64 = simd_sumcheck_inplace(buf, |s0, s1| {
        let s0_ef: EF = unsafe { core::mem::transmute_copy(&s0) };
        let s1_ef: EF = unsafe { core::mem::transmute_copy(&s1) };
        transcript.write(s0_ef);
        transcript.write(s1_ef);
        let chg_ef: EF = transcript.read();
        unsafe { core::mem::transmute_copy(&chg_ef) }
    });

    // Cast Sumcheck<F64> → Sumcheck<EF>.
    let result: Sumcheck<EF> = Sumcheck {
        verifier_messages: unsafe {
            core::mem::transmute::<Vec<crate::tests::F64>, Vec<EF>>(result_f64.verifier_messages)
        },
        prover_messages: unsafe {
            core::mem::transmute::<Vec<(crate::tests::F64, crate::tests::F64)>, Vec<(EF, EF)>>(
                result_f64.prover_messages,
            )
        },
    };

    Some(result)
}

// ─── Double-buffered sumcheck loop ──────────────────────────────────────────

/// Double-buffered SIMD sumcheck over raw Montgomery-form `u64` values.
///
/// Pre-allocates one extra buffer of size `n/2`. Each round reads from one
/// buffer and reduces into the other. Since src/dst are non-overlapping,
/// parallel writes via `par_chunks_mut` are trivially safe.
///
/// Memory cost: one allocation of `n/2 * 8` bytes at the start. Zero per-round.
#[cfg(target_arch = "aarch64")]
fn simd_sumcheck_inplace(
    buf: &mut [u64],
    mut round_step: impl FnMut(crate::tests::F64, crate::tests::F64) -> crate::tests::F64,
) -> Sumcheck<crate::tests::F64> {
    use crate::simd_fields::SimdAccelerated;
    use crate::tests::F64;

    let n = buf.len();
    let num_rounds = n.trailing_zeros() as usize;
    let mut prover_messages: Vec<(F64, F64)> = Vec::with_capacity(num_rounds);
    let mut verifier_messages: Vec<F64> = Vec::with_capacity(num_rounds);

    // Second buffer for double-buffering (only n/2 needed).
    let mut buf_b: Vec<u64> = vec![0u64; n / 2];

    // Track which buffer holds the current data.
    let mut active_len = n;
    let mut read_from_a = true; // true = data in buf (a), false = data in buf_b (b)

    for round in 0..num_rounds {
        let half = active_len / 2;

        // ── Evaluate: sum even/odd elements from the current buffer ──
        let src = if read_from_a {
            &buf[..active_len]
        } else {
            &buf_b[..active_len]
        };
        let (s0, s1) = simd_evaluate(src);

        let msg_s0 = F64::from_raw(s0);
        let msg_s1 = F64::from_raw(s1);

        prover_messages.push((msg_s0, msg_s1));
        let challenge = round_step(msg_s0, msg_s1);
        verifier_messages.push(challenge);

        // ── Reduce: read from current buffer, write to the other ──
        if round < num_rounds - 1 {
            let c = F64::to_raw(challenge);
            if read_from_a {
                simd_reduce_double(&buf[..active_len], &mut buf_b[..half], c);
            } else {
                simd_reduce_double(&buf_b[..active_len], &mut buf[..half], c);
            }
            active_len = half;
            read_from_a = !read_from_a;
        }
    }

    Sumcheck {
        verifier_messages,
        prover_messages,
    }
}

// ─── SIMD evaluate & reduce: Montgomery ops, double-buffered ────────────────

/// Sum even-indexed and odd-indexed elements.
///
/// Each rayon task sums a 16K-pair chunk with fast Montgomery adds.
#[cfg(target_arch = "aarch64")]
fn simd_evaluate(evals: &[u64]) -> (u64, u64) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        evals
            .par_chunks(16_384 * 2)
            .map(|chunk| {
                let mut s0 = MontBackend::ZERO;
                let mut s1 = MontBackend::ZERO;
                let mut i = 0;
                while i + 1 < chunk.len() {
                    s0 = MontBackend::scalar_add(s0, chunk[i]);
                    s1 = MontBackend::scalar_add(s1, chunk[i + 1]);
                    i += 2;
                }
                (s0, s1)
            })
            .reduce(
                || (MontBackend::ZERO, MontBackend::ZERO),
                |(a0, a1), (b0, b1)| {
                    (
                        MontBackend::scalar_add(a0, b0),
                        MontBackend::scalar_add(a1, b1),
                    )
                },
            )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut s0 = MontBackend::ZERO;
        let mut s1 = MontBackend::ZERO;
        let mut i = 0;
        while i + 1 < evals.len() {
            s0 = MontBackend::scalar_add(s0, evals[i]);
            s1 = MontBackend::scalar_add(s1, evals[i + 1]);
            i += 2;
        }
        (s0, s1)
    }
}

/// Double-buffered pairwise reduce: read from `src`, write to `dst`.
///
/// `dst[i] = src[2i] + c * (src[2i+1] - src[2i])`
///
/// Since `src` and `dst` are non-overlapping slices:
/// - parallel writes to `dst` via `par_chunks_mut` are trivially safe
/// - reads from `src` are shared immutable references
/// - zero per-round allocation, no `unsafe`
#[cfg(target_arch = "aarch64")]
fn simd_reduce_double(src: &[u64], dst: &mut [u64], c: u64) {
    let half = dst.len();
    debug_assert!(src.len() >= 2 * half);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        // 16K output elements per rayon task (reads 32K input elements).
        let chunk_size = 16_384;

        dst.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, dst_chunk)| {
                let src_offset = chunk_idx * chunk_size * 2;
                for i in 0..dst_chunk.len() {
                    let a = src[src_offset + 2 * i];
                    let b = src[src_offset + 2 * i + 1];
                    let diff = MontBackend::scalar_sub(b, a);
                    let scaled = MontBackend::scalar_mul(c, diff);
                    dst_chunk[i] = MontBackend::scalar_add(a, scaled);
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..half {
            let a = src[2 * i];
            let b = src[2 * i + 1];
            let diff = MontBackend::scalar_sub(b, a);
            let scaled = MontBackend::scalar_mul(c, diff);
            dst[i] = MontBackend::scalar_add(a, scaled);
        }
    }
}
