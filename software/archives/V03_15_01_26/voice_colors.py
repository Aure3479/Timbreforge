"""voice_colors.py
================================================================================
Voice "Colour" Channels for Voice→Haptics (Vibraforge)
================================================================================

Why this file exists
--------------------
This project has two complementary ways to turn voice audio into a small number of
control channels for haptics:

1) "Normal" spectral buckets (frequency bands)
   - Divide the spectrum into M buckets (M ≈ number of actuators).
   - Each actuator corresponds to one bucket (or a small set of buckets).

2) "Colour" channels (timbre/prosody-inspired descriptors)
   - Instead of *only* splitting the spectrum evenly, we define semantically named
     channels such as "Warmth", "Presence", "Shimmer", etc.
   - Each colour is expressed as a weighted combination of the canonical spectral
     bands used by the analysis front-end.

The key design choice is that we keep the **canonical front-end** fixed
(K=32 ERB-spaced bands by default) and we define colours as a matrix that maps
canonical energies -> colour energies.

This is deliberate:
- It keeps colour definitions independent of FFT size / microphone / GUI changes.
- It makes colour mode comparable across users and across different numbers of
  physical actuators.
- It enables easy iteration: change definitions here, not the DSP plumbing.

How "colour" is represented here
--------------------------------
A colour channel is defined as:
- a name (stable identifier)
- a human-readable description
- one or more frequency ranges in Hz
- an optional per-range weight
- a UI colour (hex), used ONLY for GUI visual cues (placeholder: white)

IMPORTANT:
- These are not claim to be the "one true" definition of vocal colour.
- They are pragmatic, engineering-friendly proxy features.
- You can add/modify channels later without changing controller logic.

The default set below includes a compact 6-colour basis that matches common vocal
pedagogy descriptors, plus a few optional extras.

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import json
import math
import numpy as np


HzRange = Tuple[float, float]


@dataclass
class ColorChannel:
    """A single vocal-colour channel.

    Parameters
    ----------
    name:
        Stable identifier used in code and configuration.

    description:
        Human-readable explanation (good for tooltips / reports).

    ranges_hz:
        One or more frequency intervals that contribute to this colour.
        Example: [(180, 400)]

    weights:
        Optional list of weights aligned with ranges_hz.
        If None, each range gets weight 1.

    ui_hex:
        GUI colour (placeholder here). The haptics pipeline does not depend on it.

    edge_softness_hz:
        If > 0, we use a smooth raised-cosine taper at range edges.
        This helps avoid hard discontinuities when a singer moves slightly.

    Notes
    -----
    - The "colour" concept is implemented as a weighted sum over canonical
      spectral bands.
    - A band contributes if its center frequency falls inside one of the ranges.
    """

    name: str
    description: str
    ranges_hz: List[HzRange]
    weights: Optional[List[float]] = None
    ui_hex: str = "#FFFFFF"  # placeholder: user can set real colours later
    edge_softness_hz: float = 0.0

    def range_weights(self) -> List[float]:
        if self.weights is None:
            return [1.0] * len(self.ranges_hz)
        if len(self.weights) != len(self.ranges_hz):
            # Fail safe: if mismatch, ignore provided weights.
            return [1.0] * len(self.ranges_hz)
        return [float(w) for w in self.weights]


# -----------------------------------------------------------------------------
# Default channel sets
# -----------------------------------------------------------------------------


def default_voice_color_channels(fmin: float = 50.0, fmax: float = 12000.0) -> List[ColorChannel]:
    """Return a pragmatic default palette of vocal colour channels.

    The chosen bands align with common perceptual descriptors used in vocal
    pedagogy and timbre discussion.

    Why these 6 as a *starting* basis
    ---------------------------------
    With few actuators (≈ 6), each channel must be:
    - discriminable tactilely (not too many)
    - stable across tokens / vowels (not too fragile)
    - interpretable (usable for embodied learning)

    The 6 below correspond to broad spectral regions often associated with:
    - low harmonic energy (warmth/body)
    - vowel richness (lower-mid)
    - intelligibility/aim (presence)
    - edge/bite (upper mid)
    - high harmonic sparkle (shimmer)
    - very high overtone halo (halo)

    The exact frequency values are not absolute; they are designed to be
    *reasonable defaults* that you can calibrate per user.

    fmin/fmax are applied as clamping (channels outside the analysis range are
    still returned, but will naturally be silent if the DSP fmax is lower).
    """

    # A compact 6-colour palette (the "basis" for your current experiments)
    colors = [
        ColorChannel(
            name="warmth",
            description="Low harmonic body / warmth. Often correlates with chest resonance and intimate proximity.",
            ranges_hz=[(180.0, 400.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=50.0,
        ),
        ColorChannel(
            name="velvet",
            description="Lower-mid vowel richness (often related to F1 region and vowel openness).",
            ranges_hz=[(500.0, 900.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=80.0,
        ),
        ColorChannel(
            name="presence",
            description="Directionality / intelligibility / 'aim' of the voice (roughly 1.5–3 kHz).",
            ranges_hz=[(1500.0, 3000.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=150.0,
        ),
        ColorChannel(
            name="edge",
            description="Upper-mid 'bite' / edge. Too much can feel harsh; useful for urgency and articulation.",
            ranges_hz=[(3000.0, 5000.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=200.0,
        ),
        ColorChannel(
            name="shimmer",
            description="High harmonic shimmer / sparkle (often 6.5–8 kHz).",
            ranges_hz=[(6500.0, 8000.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=250.0,
        ),
        ColorChannel(
            name="halo",
            description="Very high overtone 'halo' / air extension (often 10–12 kHz).",
            ranges_hz=[(10000.0, 12000.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=300.0,
        ),
    ]

    # Optional extras you may want later (kept here as examples)
    # - These are NOT part of the 6-colour baseline.
    # - You can activate them if you have more actuators, or if you use time-multiplexing.
    colors += [
        ColorChannel(
            name="voicing",
            description="Fundamental/voicing energy (F0 region). Helpful for pitch stability / onset/offset awareness.",
            ranges_hz=[(fmin, min(200.0, fmax))],
            ui_hex="#FFFFFF",
            edge_softness_hz=30.0,
        ),
        ColorChannel(
            name="nasality",
            description="Nasal formant cluster / honk risk zone (very rough proxy).",
            ranges_hz=[(800.0, 1200.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=80.0,
        ),
        ColorChannel(
            name="sibilance",
            description="Sibilants/frication emphasis (proxy; overlaps with shimmer but more noise-like).",
            ranges_hz=[(5500.0, 9000.0)],
            ui_hex="#FFFFFF",
            edge_softness_hz=250.0,
        ),
    ]

    # Clamp ranges to the analysis bounds for safety
    clamped: List[ColorChannel] = []
    for c in colors:
        new_ranges: List[HzRange] = []
        for (lo, hi) in c.ranges_hz:
            lo2 = max(float(lo), float(fmin))
            hi2 = min(float(hi), float(fmax))
            if hi2 > lo2:
                new_ranges.append((lo2, hi2))
        # If nothing survives clamping, keep the channel but with empty ranges.
        # This is useful for "future-proof" configs.
        clamped.append(
            ColorChannel(
                name=c.name,
                description=c.description,
                ranges_hz=new_ranges,
                weights=c.weights,
                ui_hex=c.ui_hex,
                edge_softness_hz=c.edge_softness_hz,
            )
        )

    return clamped


# -----------------------------------------------------------------------------
# Matrix builder: canonical energies -> colour energies
# -----------------------------------------------------------------------------


def _raised_cosine_weight(x: float, lo: float, hi: float, softness: float) -> float:
    """A smooth 0..1 membership for an interval with edge taper.

    If softness == 0:
        hard membership: 1 inside [lo, hi], else 0

    If softness > 0:
        membership rises over [lo-softness, lo], stays 1 inside [lo, hi],
        then falls over [hi, hi+softness].

    This is a *pragmatic* smoothing to avoid brittle bucket edges.
    """
    if softness <= 0.0:
        return 1.0 if (x >= lo and x <= hi) else 0.0

    # Left taper
    if x < lo - softness:
        return 0.0
    if lo - softness <= x < lo:
        t = (x - (lo - softness)) / softness
        return 0.5 - 0.5 * math.cos(math.pi * t)

    # Middle
    if lo <= x <= hi:
        return 1.0

    # Right taper
    if hi < x <= hi + softness:
        t = (x - hi) / softness
        return 0.5 + 0.5 * math.cos(math.pi * t)

    return 0.0


def build_color_matrix(
    canonical_centers_hz: np.ndarray,
    channels: List[ColorChannel],
    normalize: str = "mean",
) -> np.ndarray:
    """Build C such that: colour_energies = C @ canonical_energies.

    Parameters
    ----------
    canonical_centers_hz:
        Array of length K giving the center frequency of each canonical band.
        In this project, canonical_centers_hz is typically cfg.canonical_edges_hz[1:-1].

    channels:
        List of ColorChannel.

    normalize:
        How to normalize each row so channel magnitudes are comparable.

        - "mean": divide by number of contributing canonical bands (default).
        - "sum":  no division (channel energy grows with bandwidth).
        - "none": no normalization.

    Returns
    -------
    C:
        (N_colors, K) float32 matrix.

    Notes
    -----
    - Overlaps are allowed: a canonical band may contribute to multiple colours.
    - This is intentional: vocal percepts are not orthogonal.
    """
    centers = np.asarray(canonical_centers_hz, dtype=float)
    K = int(len(centers))
    N = int(len(channels))
    C = np.zeros((N, K), dtype=np.float32)

    # Fill raw weights
    for i, ch in enumerate(channels):
        ws = ch.range_weights()
        for rng, w in zip(ch.ranges_hz, ws):
            lo, hi = float(rng[0]), float(rng[1])
            w = float(w)
            if hi <= lo:
                continue
            for k, fc in enumerate(centers):
                C[i, k] += w * _raised_cosine_weight(float(fc), lo, hi, float(ch.edge_softness_hz))

    norm = (normalize or "none").strip().lower()

    dead_channels = []
    if norm == "mean":
        for i in range(N):
            denom = float(np.sum(C[i, :] > 0.0))
            if denom > 0:
                C[i, :] /= denom
            else:
                dead_channels.append(channels[i].name)

    elif norm == "sum":
        # no scaling
        pass

    elif norm == "none":
        # no scaling
        pass

    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    # Optional diagnostics
    if dead_channels:
        print("\n⚠️  WARNING: Dead colour channels detected (no canonical bands in range):")
        for name in dead_channels:
            ch = next((c for c in channels if c.name == name), None)
            if ch:
                print(f"   • {name}: ranges {ch.ranges_hz}")
        if len(centers) > 0:
            print(f"\nCanonical band centers (Hz): {centers}")
            print(f"Canonical min/max: {centers.min():.1f}–{centers.max():.1f} Hz\n")

    return C


# -----------------------------------------------------------------------------
# Optional: JSON import/export (to edit colour definitions without Python)
# -----------------------------------------------------------------------------


def colors_to_json_dict(channels: List[ColorChannel]) -> Dict:
    return {
        "channels": [
            {
                "name": c.name,
                "description": c.description,
                "ranges_hz": [[float(a), float(b)] for (a, b) in c.ranges_hz],
                "weights": [float(w) for w in c.range_weights()] if c.weights is not None else None,
                "ui_hex": c.ui_hex,
                "edge_softness_hz": float(c.edge_softness_hz),
            }
            for c in channels
        ]
    }


def save_colors_json(path: str, channels: List[ColorChannel]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(colors_to_json_dict(channels), f, indent=2)


def load_colors_json(path: str) -> List[ColorChannel]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[ColorChannel] = []
    for item in data.get("channels", []):
        out.append(
            ColorChannel(
                name=str(item.get("name", "unnamed")),
                description=str(item.get("description", "")),
                ranges_hz=[(float(a), float(b)) for (a, b) in item.get("ranges_hz", [])],
                weights=item.get("weights", None),
                ui_hex=str(item.get("ui_hex", "#FFFFFF")),
                edge_softness_hz=float(item.get("edge_softness_hz", 0.0)),
            )
        )
    return out
