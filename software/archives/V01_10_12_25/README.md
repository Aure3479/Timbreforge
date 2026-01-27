# V01_10_12_25 — TimbreForge / Vibraforge (archived)

## Goal
Real-time PoC: **Mic → FFT → 6 buckets → intensity (0–15) → BLE → Vibraforge actuators (addr 0..5)**.

## Key ideas
- Two-layer bucketing:
  - Canonical representation: ERB filterbank (K=32)
  - Actuator layer: M=6 buckets (voice_landmark recommended)
- Safety:
  - PANIC STOP
  - dead-man failsafe
  - non-additive reset (STOP before START on parameter change)
- Windows BLE robustness: dedicated MTA thread for Bleak/WinRT.

## How to run (typical)
1) Hardware smoke test (addr 0..5)
2) Manual BLE test tool
3) Headless realtime runner
4) Minimal GUI (Start/Stop + bars + ESC panic)

## Known limits
- Baseline focused on addr 0..5
- Energy→intensity mapping is intentionally simple
- Group/address logic differs across some scripts (OK for addr 0..5, must unify for larger setups)
