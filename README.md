# TimbreForge / TimbreTouch — Voice Timbre Colours as Forearm Haptics (Built on Vibraforge)

TimbreForge is a real-time **voice → haptics** system for vocal timbre exploration and practice.
It maps the voice to **six interpretable timbre colours** — **Warmth, Velvet, Presence, Edge, Shimmer, Halo** — each driving a **fixed actuator location** on the arm/hand.

## Built on Vibraforge (hardware/firmware baseline)
**Hardware is pre-built and based on Vibraforge (Huang).**  
Firmware is used **as-is** (only minor optional edits if needed: device name/UUID).  
This repo focuses on the **software layer**: timbre-colour mapping, calibration, GUI, and Live/Record/Replay workflow.

---

## ⚠️ Before first run (Windows)
### Disable noise reducers (very important)
TimbreForge relies on spectral energy. Noise suppression / AGC can distort the spectrum and break colour activation.
- Disable Windows microphone **Enhancements** (noise suppression, AGC, echo cancellation).
- Avoid Zoom/Teams/Discord/NVIDIA Broadcast suppression in the audio chain unless you intentionally test it.

### Safety
Prototype only. Start low intensity, short sessions. Stop if discomfort/heat occurs.

---

## Shortcuts to (start here)
- **Docs (GitHub Pages):** see `/docs/index.md`
- **Windows setup:** `/docs/setup_windows.md`
- **Vibraforge baseline:** `/docs/vibraforge_baseline.md`
- **Calibration & noise:** `/docs/calibration_noise.md`
- **Iterations / archives:** `/docs/iterations.md`
- **Troubleshooting:** `/docs/troubleshooting.md`

---

## Quickstart (Windows / conda recommended)
```bash
conda create -n timbreforge python=3.10 -y
conda activate timbreforge
conda install -c conda-forge numpy pyaudio -y
pip install -r requirements.txt
python software/current/voice_haptics_gui_patched_phase1_debugtab_v4.py
```

1. Power the Vibraforge wearable (BLE) and keep it close (≤ 1–2m).
2. Run Calibrate in silence (~2s).
3. Start Live and test vowels (“aaa” vs “iii”) and pressed vs gentle phonation.
4. Use ESC = PANIC STOP anytime.

## Repo map

- software/current/ — current demo version
- archives/ — V01 / V02 / V03 archived versions (with notes + reasoning)
- tools/vibraforge_test/ — hardware sanity tests (scan BLE, channel test, duty sweep)
- firmware/vibraforge_platformio/ — PlatformIO firmware currently used (baseline Vibraforge)
- docs/ — documentation (for GitHub Pages)
- assets/ — poster + photos + screenshots


## Media
- Poster (A2): assets/poster_A2.png
- GUI screenshot: assets/Gui.png
- Wearable photo: assets/arm.jpeg

## Credits

ESILV — Institute for Future Technologies (De Vinci)
Supervisor: Xiao Xiao
Built on Vibraforge (Huang): hardware + firmware baseline