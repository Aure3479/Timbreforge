"""
bucket_classification.py — Modular Spectrogram Bucketing for Voice→Haptics
=========================================================================

Context / Goal
--------------
This module converts real-time voice audio (typically 50 Hz – 8 kHz) into a small number
of tactile control channels ("buckets") suitable for driving a wearable vibrotactile array.

The central idea is that with *few actuators* (e.g., 6), you do NOT want "nice-looking"
spectral resolution; you want *interpretable* voice cues (voicing/pitch energy, vowel shape,
presence, brightness/noise) that users can learn quickly.

Design Pattern (Two-Layer Architecture)
--------------------------------------
We separate frequency handling into TWO layers:

1) CANONICAL SPECTRAL LAYER (stable, high-resolution)
   - Build a perceptual filterbank W with K bands across [fmin, fmax]
   - Compute canonical energies e = W @ P, where P is the FFT power spectrum

   We recommend canonical_scale="erb" because ERB spacing approximates auditory filter
   bandwidths and concentrates resolution where voice information is richest.

2) ACTUATOR BUCKET LAYER (variable, low-resolution, depends on available actuators)
   - Define M tactile buckets (M = number of available actuators)
   - Merge K canonical energies into M bucket energies using an aggregation matrix A:

         b = A @ e

   A is the only part that changes when actuator count changes.
   This makes scaling "safe": the front-end stays constant; only the mapping changes.

Why Two Layers?
---------------
- Comparability: experiments comparing bucket methods or actuator counts are fair because the
  same canonical representation is used for everything.
- Stability: no constant retuning when hardware changes.
- Modularity: "mel/erb/log/linear/voice_landmark" only changes bucket definitions, not the
  whole analysis front-end.

Important Note for Haptics
--------------------------
Buckets represent *spectral energy regions*, not vibration frequency.

Typically, each actuator uses a fixed vibrotactile carrier (e.g., 120–200 Hz, depending on
hardware) and the bucket energy controls the carrier amplitude (possibly with smoothing).

So "high-frequency bucket" means "energy from 3.5–8 kHz mapped to amplitude", NOT "vibrate at 6 kHz".

Choosing bucket_method (When to use what)
-----------------------------------------
bucket_method controls how M actuator buckets are built:

A) "voice_landmark"  (RECOMMENDED when M is small: ~6–12)
   - Buckets aligned to voice-relevant regions (F0/voicing, F1, F2, presence, noise)
   - Best for vocal learning / pedagogy and tactile discriminability

B) "mel", "erb", "log", "linear"  (BASELINES / benchmarking)
   - Directly split [fmin, fmax] into M intervals using that scale
   - Useful for systematic comparisons or generic audio applications

Suggested defaults for your project
-----------------------------------
- For M=6: canonical_scale="erb", K=32, bucket_method="voice_landmark", inner_scale="erb"
- For M=9: still "voice_landmark" (refine inside formant/presence zones), OR add feature channels elsewhere
- For M>=24: consider "erb" direct bucketing for spectral painting OR keep voice_landmark but with more splits

Runtime pipeline (typical)
--------------------------
1) frame -> FFT -> power spectrum P
2) canonical energies e = W @ P               (W: K x n_bins)
3) bucket energies b = A @ e                  (A: M x K)
4) (recommended downstream) log compression + noise gate + attack/release smoothing
5) map b to actuator intensities (and spatial layout)

This module focuses on steps 2–3 (spectral buckets).
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, List, Tuple, Dict


# -------------------------------------------------------------------------
# Types: control which scales and bucket strategies are allowed
# -------------------------------------------------------------------------

Scale = Literal["mel", "erb", "log", "linear"]

# BucketMethod:
# - "voice_landmark": semantic voice mapping (recommended for few actuators)
# - others: direct scale-based splits used mainly as baselines or for generic audio
BucketMethod = Literal["mel", "erb", "log", "linear", "voice_landmark"]


# -------------------------------------------------------------------------
# Psychoacoustic scale helpers (Hz <-> Mel, Hz <-> ERB-rate)
# -------------------------------------------------------------------------

def hz_to_mel(f_hz: np.ndarray) -> np.ndarray:
    """
    Convert frequency in Hz to Mel (HTK-style).

    Why Mel exists:
      - Mel frequency warping approximates perceptual pitch spacing.
      - More resolution at low frequencies, less at high frequencies.
    """
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert Mel back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def hz_to_erb_rate(f_hz: np.ndarray) -> np.ndarray:
    """
    Convert Hz -> ERB-rate using a common Glasberg & Moore style formula.

    ERB-rate intuition:
      - ERB-rate is approximately linear with cochlear place / auditory filter index.
      - Equal steps in ERB-rate correspond roughly to equal "auditory resolution steps".
    """
    return 21.4 * np.log10(4.37e-3 * f_hz + 1.0)


def erb_rate_to_hz(erb: np.ndarray) -> np.ndarray:
    """Convert ERB-rate -> Hz (inverse of hz_to_erb_rate)."""
    return (10.0 ** (erb / 21.4) - 1.0) / 4.37e-3


# -------------------------------------------------------------------------
# Band edge generation for different scales
# -------------------------------------------------------------------------

def spaced_edges(
    fmin: float,
    fmax: float,
    n_bands: int,
    scale: Scale,
) -> np.ndarray:
    """
    Generate n_bands+2 "edge points" used to build triangular filters.

    Filterbank convention:
      edges has length (n_bands + 2)
      filter k uses:
        left   = edges[k]
        center = edges[k+1]
        right  = edges[k+2]

    Notes:
    - This creates *triangular* filters (for canonical W).
    - For actuator buckets we often use *intervals* (lo, hi) instead.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    if not (0.0 < fmin < fmax):
        raise ValueError("Require 0 < fmin < fmax")

    if scale == "mel":
        x0, x1 = hz_to_mel(np.array([fmin, fmax], dtype=float))
        edges = np.linspace(x0, x1, n_bands + 2)
        return mel_to_hz(edges)

    if scale == "erb":
        x0, x1 = hz_to_erb_rate(np.array([fmin, fmax], dtype=float))
        edges = np.linspace(x0, x1, n_bands + 2)
        return erb_rate_to_hz(edges)

    if scale == "log":
        # Geometric spacing: equal ratios (octave-like / constant-Q-ish)
        return np.geomspace(fmin, fmax, n_bands + 2)

    if scale == "linear":
        return np.linspace(fmin, fmax, n_bands + 2)

    raise ValueError(f"Unknown scale: {scale}")


# -------------------------------------------------------------------------
# FFT-domain triangular filterbank builder
# -------------------------------------------------------------------------

def triangular_filterbank(
    sr: int,
    n_fft: int,
    edges_hz: np.ndarray,
) -> np.ndarray:
    """
    Build a triangular filterbank matrix W of shape (n_filters, n_fft_bins).

    Definitions:
    - n_fft_bins = n_fft//2 + 1 (real FFT bins, 0..Nyquist)
    - freqs are linearly spaced bin center frequencies
    - W[k, :] is a triangle spanning [left, center, right] from edges_hz

    Usage:
      P: power spectrum, shape (n_bins,)
      e: band energies, shape (n_filters,)
      e = W @ P

    Implementation details:
    - Each triangle is normalized by its sum so energies are comparable across bands.
    - If an edge triplet collapses (numerical issue), the filter is left as zeros.
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)

    n_filters = len(edges_hz) - 2
    if n_filters < 1:
        raise ValueError("Need at least 3 edge points to form 1 triangular filter.")

    W = np.zeros((n_filters, n_bins), dtype=np.float32)

    for k in range(n_filters):
        left, center, right = edges_hz[k], edges_hz[k + 1], edges_hz[k + 2]

        # Numerical safety: clamp to [0, Nyquist]
        left = max(left, 0.0)
        right = min(right, sr / 2.0)
        if not (left < center < right):
            continue

        # Rising slope: left -> center
        left_mask = (freqs >= left) & (freqs <= center)
        W[k, left_mask] = (freqs[left_mask] - left) / (center - left)

        # Falling slope: center -> right
        right_mask = (freqs >= center) & (freqs <= right)
        W[k, right_mask] = (right - freqs[right_mask]) / (right - center)

        # Normalize so each band contributes comparable magnitude
        s = W[k].sum()
        if s > 1e-9:
            W[k] /= s

    return W


# -------------------------------------------------------------------------
# Voice-landmark regions and safe scaling rules
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """
    A semantic frequency region tied to voice acoustics.

    name:   readable tag (useful for debugging/reporting)
    f0,f1:  lower and upper boundary in Hz
    weight: if M increases, regions with higher weight get more sub-buckets
    """
    name: str
    f0: float
    f1: float
    weight: float


def default_voice_regions(fmin: float, fmax: float) -> List[Region]:
    """
    Semantic regions anchored to vocal landmarks.
    Designed for 50 Hz – 8 kHz but clamps safely if fmin/fmax differ.

    These boundaries are intentionally "voice-meaningful", not purely psychoacoustic.

    Rationale:
    - voicing_F0: captures voiced energy / pitch power region
    - low_harmonics: warmth/chest energy
    - F1_openness: vowel openness correlates strongly with F1 band energy shifts
    - F2_articulation: vowel front/back and articulation correlate strongly with F2
    - presence_ring: "presence/projection" area (roughly 2.5–3.5 kHz)
    - brightness_noise: breath/frication/sibilance + "air" region
    """
    regions = [
        Region("voicing_F0",          50.0,  200.0, 1.0),
        Region("low_harmonics",      200.0,  400.0, 1.0),
        Region("F1_openness",        400.0,  900.0, 2.0),
        Region("F2_articulation",    900.0, 2500.0, 3.0),
        Region("presence_ring",     2500.0, 3500.0, 1.5),
        Region("brightness_noise",  3500.0, 8000.0, 1.0),
    ]

    # Clamp to requested [fmin, fmax]
    clamped: List[Region] = []
    for r in regions:
        lo = max(r.f0, fmin)
        hi = min(r.f1, fmax)
        if hi > lo:
            clamped.append(Region(r.name, lo, hi, r.weight))
    return clamped


def allocate_buckets_across_regions(regions: List[Region], M: int) -> Dict[int, int]:
    """
    Decide how many buckets each region gets, given total M buckets.

    Returns:
      dict: region_index -> n_buckets_in_that_region

    Strategy:
    - If M >= number of regions:
        each region gets 1 bucket minimum, then distribute remaining buckets
        proportional to region weights (more detail in formant zones).
    - If M < number of regions:
        keep the lowest M regions, and merge all remaining higher regions into
        the last active bucket later (safe compression for few tactors).
    """
    R = len(regions)
    if M < 1:
        raise ValueError("M must be >= 1")
    if R == 0:
        raise ValueError("No regions available (check fmin/fmax).")

    # Case 1: Enough buckets to give every region at least 1
    if M >= R:
        alloc = {i: 1 for i in range(R)}
        extra = M - R
        if extra == 0:
            return alloc

        weights = np.array([regions[i].weight for i in range(R)], dtype=float)
        weights = weights / (weights.sum() + 1e-12)

        # Distribute extra buckets with rounding, then fix sum by greedy remainder
        extra_float = weights * extra
        extra_int = np.floor(extra_float).astype(int)
        remainder = extra - int(extra_int.sum())

        frac = extra_float - extra_int
        order = np.argsort(-frac)
        for j in range(remainder):
            extra_int[order[j]] += 1

        for i in range(R):
            alloc[i] += int(extra_int[i])
        return alloc

    # Case 2: Fewer buckets than regions => preserve low regions first.
    alloc = {i: 0 for i in range(R)}
    for i in range(M):
        alloc[i] = 1
    return alloc


def split_region_into_edges(
    fmin: float,
    fmax: float,
    n: int,
    inner_scale: Scale,
) -> List[Tuple[float, float]]:
    """
    Split one region [fmin, fmax] into n contiguous sub-buckets.

    Returns:
      list of (lo, hi) intervals, length n

    inner_scale determines how we place split points:
    - "erb"/"mel" for perceptual spacing inside semantic regions
    - "log" for ratio spacing
    - "linear" for equal Hz

    Note:
    - This produces *interval edges* (lo, hi), not triangular filters.
    """
    if n <= 0:
        return []
    if n == 1:
        return [(fmin, fmax)]

    if inner_scale == "mel":
        e = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n + 1)
        hz = mel_to_hz(e)
    elif inner_scale == "erb":
        e = np.linspace(hz_to_erb_rate(np.array([fmin]))[0], hz_to_erb_rate(np.array([fmax]))[0], n + 1)
        hz = erb_rate_to_hz(e)
    elif inner_scale == "log":
        hz = np.geomspace(fmin, fmax, n + 1)
    elif inner_scale == "linear":
        hz = np.linspace(fmin, fmax, n + 1)
    else:
        raise ValueError(inner_scale)

    out: List[Tuple[float, float]] = []
    for i in range(n):
        lo, hi = float(hz[i]), float(hz[i + 1])
        if hi > lo:
            out.append((lo, hi))
    return out


# -------------------------------------------------------------------------
# The main generator: builds canonical bank + actuator bucket mapping
# -------------------------------------------------------------------------

@dataclass
class BucketConfig:
    """
    Output configuration used at runtime.

    canonical_edges_hz:
      (K+2,) edge points for canonical triangular filterbank creation

    canonical_W:
      (K, n_bins) matrix mapping FFT power spectrum -> canonical energies e (size K)

    bucket_edges_hz:
      list of M (lo, hi) ranges describing the actuator bucket frequency intervals

    A:
      (M, K) aggregation matrix mapping canonical energies -> actuator bucket energies b (size M)

    sr, n_fft:
      needed to interpret FFT bins / frequencies

    fmin, fmax:
      analysis bounds (50–8000 voice default)

    K, M:
      canonical bands K, actuator buckets M

    canonical_scale:
      ERB/Mel/log/linear spacing for the canonical filterbank (recommended ERB)

    bucket_method:
      voice_landmark (semantic) or direct scale-based bucketing (baselines)
    """
    canonical_edges_hz: np.ndarray
    canonical_W: np.ndarray

    bucket_edges_hz: List[Tuple[float, float]]
    A: np.ndarray

    sr: int
    n_fft: int
    fmin: float
    fmax: float
    K: int
    M: int
    canonical_scale: Scale
    bucket_method: BucketMethod


def make_bucket_config(
    sr: int,
    n_fft: int,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    # Canonical representation: compute many bands once, then merge down
    K: int = 32,
    canonical_scale: Scale = "erb",
    # Output: number of actuators / buckets (depends on hardware availability)
    M: int = 6,
    # How to choose buckets for actuators
    bucket_method: BucketMethod = "voice_landmark",
    # If voice_landmark splits regions internally, which spacing to use inside each region
    inner_scale: Scale = "erb",
) -> BucketConfig:
    """
    Build a modular frequency bucketing configuration.

    Parameters (what to set, what values, why)
    -----------------------------------------
    sr (int):
      Sample rate (Hz). Must satisfy fmax <= sr/2 (Nyquist).
      Typical: 48000 or 44100.

    n_fft (int):
      FFT size. Larger => better frequency resolution, higher latency/CPU.
      Typical for voice: 1024 or 2048 at 48kHz.

    fmin, fmax (float):
      Frequency analysis range. Voice default is 50..8000 Hz.
      fmin below ~50 is typically less useful for voice (and may require special AM tricks).
      fmax above ~8k adds more noise/less pedagogical value for vocal feedback.

    K (int):
      Canonical band count. Recommended 24..48, commonly 32.
      K is independent of actuator count and should remain stable across experiments.

    canonical_scale (Scale):
      How canonical band edges are spaced. Recommended "erb" (auditory filter bandwidth-like).

    M (int):
      Number of actuator channels (available tactors). Examples: 6, 9, 24, 32, 128.

    bucket_method (BucketMethod):
      - "voice_landmark": semantic regions aligned to voice cues (best for small M)
      - "mel"/"erb"/"log"/"linear": direct spacing baselines

    inner_scale (Scale):
      Only used by "voice_landmark" when a region is split into multiple buckets.
      Recommended "erb" for perceptual spacing within each semantic region.

    Returns
    -------
    BucketConfig with:
      canonical_W for e = W @ P
      A for b = A @ e
      and human-readable bucket_edges_hz for debugging/UI.

    Runtime summary
    ---------------
      P = |rfft(frame)|^2
      e = canonical_W @ P          (size K)
      b = A @ e                    (size M)
    """
    if fmax > sr / 2:
        raise ValueError("fmax must be <= Nyquist (sr/2). Increase sr or lower fmax.")
    if M < 1:
        raise ValueError("M must be >= 1")
    if K < 1:
        raise ValueError("K must be >= 1")

    # 1) Canonical filterbank (stable across all actuator counts)
    canonical_edges_hz = spaced_edges(fmin, fmax, K, canonical_scale)
    canonical_W = triangular_filterbank(sr=sr, n_fft=n_fft, edges_hz=canonical_edges_hz)

    # 2) Choose bucket edges (M contiguous intervals)
    bucket_edges: List[Tuple[float, float]] = []

    if bucket_method in ("mel", "erb", "log", "linear"):
        # Direct baselines:
        # Build M buckets by splitting [fmin,fmax] into M intervals on the chosen scale.
        # This makes systematic comparisons easy, but may be less interpretable for voice pedagogy.
        scale = bucket_method  # type: ignore

        if scale == "mel":
            e = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], M + 1)
            hz = mel_to_hz(e)
        elif scale == "erb":
            e = np.linspace(hz_to_erb_rate(np.array([fmin]))[0], hz_to_erb_rate(np.array([fmax]))[0], M + 1)
            hz = erb_rate_to_hz(e)
        elif scale == "log":
            hz = np.geomspace(fmin, fmax, M + 1)
        elif scale == "linear":
            hz = np.linspace(fmin, fmax, M + 1)
        else:
            raise ValueError(scale)

        for i in range(M):
            lo, hi = float(hz[i]), float(hz[i + 1])
            if hi > lo:
                bucket_edges.append((lo, hi))

        # If numerical issues reduce count, pad with last bucket
        if len(bucket_edges) != M:
            if len(bucket_edges) == 0:
                bucket_edges = [(fmin, fmax)] * M
            else:
                while len(bucket_edges) < M:
                    bucket_edges.append(bucket_edges[-1])

    elif bucket_method == "voice_landmark":
        # Semantic (recommended for few actuators):
        # Build buckets based on voice landmark regions, then split regions if M is larger.
        regions = default_voice_regions(fmin, fmax)
        alloc = allocate_buckets_across_regions(regions, M)

        # Active regions are those allocated >= 1 bucket.
        active_region_indices = [i for i in range(len(regions)) if alloc.get(i, 0) > 0]

        if len(active_region_indices) == 0:
            bucket_edges = [(fmin, fmax)] * M
        else:
            # Build buckets for active regions
            for i in active_region_indices:
                r = regions[i]
                n = alloc[i]
                bucket_edges.extend(split_region_into_edges(r.f0, r.f1, n, inner_scale))

            # Merge leftover high regions into the last bucket if M < number of regions
            if len(active_region_indices) < len(regions):
                last_lo, last_hi = bucket_edges[-1]
                for j in range(active_region_indices[-1] + 1, len(regions)):
                    last_hi = max(last_hi, regions[j].f1)
                bucket_edges[-1] = (last_lo, min(last_hi, fmax))

            # Enforce exactly M buckets
            if len(bucket_edges) > M:
                merged_lo = bucket_edges[M - 1][0]
                merged_hi = bucket_edges[-1][1]
                bucket_edges = bucket_edges[: M - 1] + [(merged_lo, merged_hi)]
            while len(bucket_edges) < M:
                bucket_edges.append(bucket_edges[-1])

    else:
        raise ValueError(f"Unknown bucket_method: {bucket_method}")

    # 3) Build aggregation matrix A: map canonical bands -> bucket bands
    #
    # Approach:
    # - Each canonical band has a "center frequency" fc.
    # - Assign each canonical fc to exactly one bucket interval (lo, hi).
    # - Then A[bucket, canonical] = 1 for assigned pairs.
    # - Finally normalize each row to average (so bucket energies are scale-stable).
    #
    # This keeps behavior stable as M changes, because canonical band energies are unchanged.
    #
    K_actual = canonical_W.shape[0]
    canonical_centers = canonical_edges_hz[1:-1]  # length K
    A = np.zeros((M, K_actual), dtype=np.float32)
    bucket_counts = np.zeros(M, dtype=np.int32)

    for k, fc in enumerate(canonical_centers):
        assigned = None

        # Find bucket where lo <= fc < hi (last bucket includes fc==hi)
        for i, (lo, hi) in enumerate(bucket_edges):
            if (fc >= lo and fc < hi) or (i == M - 1 and fc <= hi):
                assigned = i
                break

        if assigned is None:
            # Clamp out-of-range due to numerical edge effects
            assigned = 0 if fc < bucket_edges[0][0] else (M - 1)

        A[assigned, k] = 1.0
        bucket_counts[assigned] += 1

    # Normalize each bucket row to average the canonical bands assigned to it
    for i in range(M):
        if bucket_counts[i] > 0:
            A[i, :] /= float(bucket_counts[i])
        else:
            # Rare corner case: if no canonical centers fall in a bucket,
            # copy the closest non-empty bucket to avoid dead channels.
            non_empty = np.where(bucket_counts > 0)[0]
            if len(non_empty) > 0:
                j = int(non_empty[np.argmin(np.abs(non_empty - i))])
                A[i, :] = A[j, :]

    return BucketConfig(
        canonical_edges_hz=canonical_edges_hz,
        canonical_W=canonical_W,
        bucket_edges_hz=bucket_edges,
        A=A,
        sr=sr,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        K=K_actual,
        M=M,
        canonical_scale=canonical_scale,
        bucket_method=bucket_method,
    )


# -------------------------------------------------------------------------
# Example runtime usage (core math only)
# -------------------------------------------------------------------------

def compute_bucket_energies(frame: np.ndarray, cfg: BucketConfig) -> np.ndarray:
    """
    Given a single audio frame (time-domain), compute bucket energies (size M).

    IMPORTANT:
    - In real-time you should window the frame (e.g. Hann) and manage overlap externally.
    - For tactile comfort & clarity, apply downstream:
        * log compression
        * noise gate
        * attack/release smoothing
    - This function is intentionally minimal: it demonstrates the math for W and A.

    Steps:
      1) FFT -> power spectrum P
      2) canonical energies e = W @ P
      3) bucket energies b = A @ e
    """
    X = np.fft.rfft(frame, n=cfg.n_fft)
    P = (np.abs(X) ** 2).astype(np.float32)

    e = cfg.canonical_W @ P   # shape (K,)
    b = cfg.A @ e             # shape (M,)

    return b


# -------------------------------------------------------------------------
# Quick reference: suggested configs (paste into your project docs if desired)
# -------------------------------------------------------------------------

"""
CHEAT SHEET — Recommended settings

For your current PoC (M=6 working actuators):
    cfg = make_bucket_config(
        sr=48000,
        n_fft=2048,
        fmin=50.0,
        fmax=8000.0,
        K=32,
        canonical_scale="erb",      # stable, perceptual front-end
        M=6,
        bucket_method="voice_landmark",  # best for learnability with few tactors
        inner_scale="erb",
    )

For baseline comparisons (same canonical front-end, different bucketing):
    cfg_mel = make_bucket_config(..., M=6, bucket_method="mel")
    cfg_log = make_bucket_config(..., M=6, bucket_method="log")
    cfg_lin = make_bucket_config(..., M=6, bucket_method="linear")

Scaling:
- Keep K and canonical_scale fixed across conditions.
- Only change M and bucket_method (and maybe inner_scale).
"""


if __name__ == "__main__":
    # Demo: build a 6-actuator voice-landmark config and print bucket ranges
    cfg6 = make_bucket_config(
        sr=48000,
        n_fft=2048,
        fmin=50.0,
        fmax=8000.0,
        K=32,
        canonical_scale="erb",
        M=6,
        bucket_method="voice_landmark",
        inner_scale="erb",
    )
    print("Bucket edges (Hz):")
    for i, (lo, hi) in enumerate(cfg6.bucket_edges_hz):
        print(f"  bucket[{i}]: {lo:.1f} – {hi:.1f} Hz")
