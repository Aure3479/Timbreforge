# V02_04_01_26 — TimbreForge / Vibraforge (archived)

## Goal
Same end-to-end pipeline as V1, but adds a **tuning-oriented “product layer”**:
GUI controls, presets, RMS monitoring, more robust energy→duty mapping, and stronger Windows BLE stability.

## What changed vs V1
- GUI: per-bucket gain/frequency, noise gate adjustments, presets
- Controller: loudness factor + spectral contrast shaping (more perceptual)
- Monitoring: RMS mean/peak over time windows
- Robustness: safety patterns + MTA thread for BLE ops.

## Known issue to document
Addressing logic may differ between low-level test scripts and controller for higher addresses; baseline 0..5 unaffected.
