# Vibraforge Test Tool (BLE + Actuators) — single-file CLI

This folder contains **one** low-level test script:

- `vibraforge_test.py`

It is used to validate **Vibraforge hardware/firmware** (BLE connection + actuator response)
**without** running the full TimbreForge/TimbreTouch audio pipeline.

---

## Built on Vibraforge (hardware/firmware baseline)

**Important:** TimbreForge/TimbreTouch reuses **Vibraforge hardware + firmware** (Huang) as the baseline.
This test tool only sends **manual BLE commands** to the existing Vibraforge firmware.

- Hardware is **pre-built** (wearable + actuators + wiring).
- Firmware is assumed **as-is** (PlatformIO project lives elsewhere in this repo).
- If anything doesn’t respond, first suspect: **BLE name/UUID mismatch** or **address mapping**, not TimbreForge.

---

## ⚠️ Safety first (read before running)

This script can trigger repeated vibrations across actuators.

- Start with **low duty** (e.g., 2–6) and short durations.
- Keep an easy way to stop: choose **mode=0** (stop) in manual mode if needed.
- If anything feels uncomfortable, stop immediately and power off the wearable.
- Do not use on irritated skin. Avoid tight straps.

**Prototype disclaimer:** not a medical device.

---

## Requirements (Windows)

- Windows 10/11 with Bluetooth enabled
- Python environment with:
  - `bleak`

Recommended (use your main TimbreForge conda env):

```bash
conda activate timbreforge
pip install bleak
```


# VibraForge Test Tool (BLE + Actuators) — `tools/vibraforge_test.py`

This tool lets you sanity-check your VibraForge wearable (BLE + actuators) **before** debugging the TimbreForge audio pipeline.

---

## Run

From the repo root:

```bash
python tools/vibraforge_test.py
```

##If your BLE device name or UUID differs

Override via CLI flags:

```bash
python tools/vibraforge_test.py -name "QT Py ESP32-S3" -uuid "f22535de-5375-44bd-8ca9-d0ea9ff9e410"
```

These values must match what your firmware advertises.

# Menu guide (interactive)

When connected, you will see:

## 1 — Manual test

Enter: addr, duty, freq, and start/stop (1 to start, 0 to stop)
If you start, the script waits ~2s then sends a stop signal.

## 2 — Single address sweep

Pick one addr, choose a freq
The script sweeps duty = 0..15
After each step, you can continue or stop.

## 3 — Full address sweep


Choose start_addr, end_addr, freq, duration
For each address, it sweeps duty = 0..15


# Protocol details (as implemented in vibraforge_test.py)

The script documents a 3-byte command format and builds the bytes like this:

```bash
serial_group = addr // 16
serial_addr  = addr % 16
```

Then:
```bash
byte1 = (serial_group << 2) | (mode & 0x01)
byte2 = 0x40 | (serial_addr & 0x3F)               # 0x40 encodes the leading 01
byte3 = 0x80 | ((duty & 0x0F) << 3) | freq         # 0x80 encodes the leading 1
```

The script sends one command, then pads the packet with 0xFF 0xFF 0xFF repeated 19 times, so the write is:

**60 bytes total** = (20 commands × 3 bytes)

**Stop signal**: mode=0 and duty=0 (with a frequency value still provided).

Note: a comment mentions a wave bit, but the current implementation only packs duty and freq.
If your firmware expects a different packing (e.g., freq+wave), this is the first place to adjust.

# Common pitfalls / troubleshooting
## “Device not found”

- The script matches device name exactly. Windows sometimes reports no name (None) or a different name.
- Fix: run again, watch scan output, then pass -name "...".
- Ensure the board is powered and advertising.
- Move closer (≤ 1–2m).

## “Connected, but no vibration”

- Wrong UUID or wrong characteristic (use -uuid).
- Wrong address mapping (some setups don’t use addr//16 grouping).
- Duty too low: try duty 8–12 briefly.
- Your firmware might require a different packing for byte3 (freq/wave variant).

## “It vibrates but STOP doesn’t work”

- The script stops by sending mode=0 with duty=0 on the same address.
- If your firmware expects a different stop semantic, adjust send_stop_signal() in the script.

## “Windows BLE is unstable”
- Toggle Bluetooth off/on, restart the board, relaunch the script.
- Close other BLE-heavy apps.


## Relationship to TimbreForge / TimbreTouch

Use this test tool to confirm the wearable is working before debugging the audio pipeline.

Recommended debug order:

1) tools/vibraforge_test.py (BLE + actuators sanity)
2) TimbreForge GUI (Live/Record/Replay + calibration)
3) Only then tweak mapping / noise floor / thresholds

## Disclaimers about noise reducers (main project)

This test tool doesn’t use audio, but for TimbreForge itself:
- Disable Windows mic “Enhancements” (noise suppression, AGC, echo cancellation)
- Avoid Zoom/Teams/Discord/NVIDIA Broadcast suppression unless you want to test that effect deliberately
- TimbreForge uses ambient-noise calibration + noise-floor subtraction, not aggressive denoising