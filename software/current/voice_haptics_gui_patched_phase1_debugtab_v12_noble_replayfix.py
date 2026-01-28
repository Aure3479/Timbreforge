# voice_haptics_gui.py
"""================================================================================
Vibraforge – GUI (Voice → Haptics) Control Panel
================================================================================

This GUI drives the real-time voice→haptics pipeline via a worker thread.

What the GUI provides
---------------------
- Start/Stop real-time processing.
- ESC = PANIC STOP (immediate motor stop, without closing the app).
- Live actuator intensity display (0..15).
- Per-actuator vibrotactile frequency index (0..7) and gain.
- Global noise gate (dB).
- Two processing modes:

  1) Normal mode (spectral buckets)
     - You define M frequency buckets (M = number of available actuators).
     - Energy in each bucket drives the corresponding actuator.
     - Buckets can be edited live (boundaries + optional overlap).
     - Built-in templates (voice-landmark / ERB / Mel / Log / Linear).

  2) Colour mode (timbre “colours”)
     - The analysis front-end stays canonical (K=32 ERB bands).
     - We compute a small set of named “colour channels” (warmth/presence/etc).
     - Each actuator can be assigned a colour (multiple actuators can share one).
     - This lets you prototype expressive layouts even with few actuators.

How it talks to the controller
------------------------------
- Qt runs in the main thread.
- A HapticsWorker lives in a QThread and creates its own asyncio loop.
- The worker owns a VoiceToHapticsController instance.
- GUI changes are forwarded through thread-safe callbacks that run in the
  worker’s asyncio loop.

Why a worker thread + asyncio loop?
-----------------------------------
- PyQt5’s event loop (GUI) should stay responsive.
- Bleak (BLE) uses asyncio.
- This architecture prevents BLE/audio blocking the UI.

Note
----
If you change the number of available actuators, set N_ACTUATORS below and
restart the GUI.
================================================================================
"""

from __future__ import annotations

import asyncio
import sys
import time
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QFileDialog,
    QLineEdit,
    QCheckBox,
    QProgressBar,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSlider,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from bucket_classification import make_bucket_config
from voice_colors import default_voice_color_channels
from voice_to_haptics_controller_patched_phase1_v12_noble_replayfix import VoiceToHapticsController


# -----------------------------------------------------------------------------
# GLOBAL GUI CONSTANTS
# -----------------------------------------------------------------------------

# Your current prototype: "9 actuators readily available, 6 in reality".
# The UI is written to be easily adjustable: change N_ACTUATORS here.
N_ACTUATORS = 6

ADDRS_PER_CHANNEL = 30
MAX_CHANNELS = 8


# Analysis defaults (must match the controller defaults if you want templates
# to align perfectly).
DEFAULT_SR = 48000
DEFAULT_N_FFT = 2048
DEFAULT_FMIN = 50.0
DEFAULT_FMAX = 12000.0
DEFAULT_K = 32

# Color palette for intensity visualization (0-15)
INTENSITY_PALETTE = [
    '#000000', # 0: Noir
    '#8A2BE2', # 1: Violet
    '#C71585', # 2: Rose fuchsia
    '#FF1493', # 3: Rose vif
    '#DC143C', # 4: Rouge cramoisi
    '#FF4500', # 5: Orange rouge
    '#FF8C00', # 6: Orange foncé
    '#FFD700', # 7: Jaune or
    '#ADFF2F', # 8: Vert lime
    '#32CD32', # 9: Vert moyen
    '#00FF7F', # 10: Vert printemps
    '#00FFFF', # 11: Cyan pur
    '#00BFFF', # 12: Bleu ciel
    '#1E90FF', # 13: Bleu dodger
    '#0000FF', # 14: Bleu pur
    '#FFFFFF'  # 15: Blanc
]

# -----------------------------------------------------------------------------
# Small helpers for display
# -----------------------------------------------------------------------------
import pyaudio

pa = pyaudio.PyAudio()
info = pa.get_default_input_device_info()
print("DEFAULT INPUT DEVICE:")
print("  index:", info["index"])
print("  name :", info["name"])
print("  chans:", info["maxInputChannels"])
print("  rate :", info["defaultSampleRate"])

print("INPUT DEVICES:")
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d.get("maxInputChannels", 0) > 0:
        print(f"[{i}] {d['name']} (ch={d['maxInputChannels']}, rate={d['defaultSampleRate']})")

def _find_input_device_index(self, name_contains: str) -> int | None:
    s = name_contains.lower()
    for i in range(self._pyaudio.get_device_count()):
        d = self._pyaudio.get_device_info_by_index(i)
        if d.get("maxInputChannels", 0) > 0 and s in d.get("name", "").lower():
            return i
    return None

def open_microphone(self, device_name_contains: str | None = None) -> None:
    if self._pyaudio is None:
        self._pyaudio = pyaudio.PyAudio()

    if self.stream is not None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass
        self.stream = None

    device_index = None
    if device_name_contains:
        device_index = self._find_input_device_index(device_name_contains)
        print("Selected input device index:", device_index, "for query:", device_name_contains)

    # Print what PortAudio considers the default (useful when device_index=None)
    try:
        dinfo = self._pyaudio.get_default_input_device_info()
        print("Default input:", dinfo["index"], dinfo["name"])
    except Exception:
        pass

    self.stream = self._pyaudio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=self.sr,
        input=True,
        input_device_index=device_index if device_index is not None else None,
        frames_per_buffer=self.chunk_size,
    )



def bar(v: int, vmax: int = 15, width: int = 18) -> str:
    """ASCII bar for 0..vmax."""
    v = int(max(0, min(v, vmax)))
    filled = int(round((v / vmax) * width))
    return "█" * filled + "░" * (width - filled)


def fmt_hz(lo: float, hi: float) -> str:
    """Pretty frequency label."""
    return f"{lo:.0f}–{hi:.0f} Hz"


def intensity_to_ui_hex(_duty: int) -> str:
    """Pipeline placeholder: map intensity to a GUI colour.

    The user asked for a pipeline where each intensity could be represented by
    a colour in the GUI. We provide a hook function here.

    For now: everything is white.

    Later: you can implement a gradient (e.g., black→blue→white) or use the
    semantic colour’s ui_hex for the background.
    """
    return "#FFFFFF"


# -----------------------------------------------------------------------------
# Worker thread: runs BLE + audio processing without blocking Qt
# -----------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """Configuration snapshot passed from GUI -> worker at start."""

    mode: str
    noise_gate_db: float
    spectral_span_db: float
    dominance_enabled: bool
    dominance_top_k: int
    dominance_group_size: int
    dominance_non_dom_gain: float
    global_freq_index: int

    # Normal mode bucket edges (Hz)
    bucket_edges: List[Tuple[float, float]]

    # Colour mode
    actuator_colour_names: List[str]

    # Per-actuator controls
    actuator_freq_indices: List[int]
    actuator_gains: List[float]
    actuator_min_duty: List[int]
    actuator_max_duty: List[int]

    actuator_channels: List[int]
    actuator_addresses: List[int]

    input_device_query: Optional[str]

    # Audio source
    audio_mode: str  # 'live' | 'record' | 'replay'
    record_path: Optional[str]
    playback_path: Optional[str]
    playback_to_speaker: bool

    # BLE / Hardware
    ble_mode: str  # 'required' | 'optional' | 'disabled'

    # Debug UI
    debug_bars_enabled: bool

    # Colour-mode noise floor subtraction
    colour_noise_subtract_enabled: bool
    colour_noise_subtract_alpha: float

class HapticsWorker(QObject):
    intensity_updated = pyqtSignal(object)  # array-like length N_ACTUATORS
    rms_updated = pyqtSignal(float, float, float, float)
    # canonical (K,), colours_raw (C,), colours_calibrated (C,), rms_db
    features_updated = pyqtSignal(object, object, object, float)
    debug_meta = pyqtSignal(object)  # dict (centers, colour_names)
    calibration_done = pyqtSignal(float, object)  # (noise_gate_db, noise_floor_colour)
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, cfg: RuntimeConfig):
        super().__init__()
        self.cfg = cfg
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.controller: Optional[VoiceToHapticsController] = None
        self.current_input_device_query = None
        self.current_audio_mode = "live"
        self._last_debug_emit_t = 0.0
        self._debug_fps = 20.0
        self._debug_enabled = bool(cfg.debug_bars_enabled)


    # ---- Thread-safe commands from GUI (these schedule work inside asyncio loop) ----

    def stop(self, immediate_panic: bool = True) -> None:
        self._running = False
        if immediate_panic:
            self.panic_stop()

    def panic_stop(self) -> None:
        """Immediate stop, safe to call from GUI thread.

        In visual-only / no-BLE mode, this still clears the GUI intensities immediately.
        """
        try:
            # Always clear the visual "haptics" output
            self.intensity_updated.emit(np.zeros(N_ACTUATORS, dtype=np.uint8))
        except Exception:
            pass

        try:
            if self._loop and self.controller and self.controller.is_connected:
                asyncio.run_coroutine_threadsafe(self.controller.panic_stop(), self._loop)
        except Exception:
            pass

    def update_mode(self, mode: str) -> None:
        if not (self._loop and self.controller):
            return

        # GUI uses "colour", controller expects "color"
        mode_norm = str(mode).strip().lower()
        if mode_norm == "colour":
            mode_norm = "color"

        def _do():
            try:
                self.controller.set_mode(mode_norm)
            except Exception:
                # Optionnel: afficher l'erreur au lieu de la masquer
                # self.status.emit(f"Mode change failed: {e}")
                pass

        self._loop.call_soon_threadsafe(_do)


    def update_noise_gate(self, noise_gate_db: float) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_noise_gate(noise_gate_db)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_spectral_span_db(self, spectral_span_db: float) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_spectral_span_db(float(spectral_span_db))
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_colour_dominance(self, enabled: bool, top_k: int, group_size: int, non_dom_gain: float) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_colour_dominance(bool(enabled), int(top_k), int(group_size), float(non_dom_gain))
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_debug_enabled(self, enabled: bool) -> None:
        self._debug_enabled = bool(enabled)

    def update_colour_noise_subtract(self, enabled: bool, alpha: float) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_colour_noise_subtract(bool(enabled), float(alpha))
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_audio_source(
        self,
        audio_mode: str,
        input_device_query: Optional[str],
        record_path: Optional[str],
        playback_path: Optional[str],
        playback_to_speaker: bool,
    ) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.open_audio_source(
                    audio_mode,
                    input_device_query=input_device_query,
                    record_path=record_path,
                    playback_path=playback_path,
                    playback_to_speaker=playback_to_speaker,
                )
                self.status.emit(f"Audio source switched -> {audio_mode}")
                # Emit meta once after switching
                if self._debug_enabled:
                    try:
                        self.debug_meta.emit(self.controller.get_debug_meta())
                    except Exception:
                        pass
            except Exception as e:
                self.status.emit(f"Audio source error: {e}")

        self._loop.call_soon_threadsafe(_do)

    def request_calibration(
        self,
        duration_s: float = 2.0,
        percentile: float = 70.0,
        gate_margin_db: float = 5.0,
        colour_percentile: float = 60.0,
    ) -> None:
        if not (self._loop and self.controller):
            return

        async def _cal():
            try:
                self.status.emit("Calibration: stay silent… (2s)")
                # Ensure haptics are stopped during capture
                try:
                    await self.controller.panic_stop()
                except Exception:
                    pass
                res = self.controller.calibrate_room_noise(
                    duration_s=float(duration_s),
                    percentile=float(percentile),
                    gate_margin_db=float(gate_margin_db),
                    colour_percentile=float(colour_percentile),
                )
                ng = float(res["noise_gate_db"])
                floor = res["noise_floor_colour"]
                self.calibration_done.emit(ng, floor)
                self.status.emit(f"Calibration done. noise_gate={ng:.1f} dB")
            except Exception as e:
                self.status.emit(f"Calibration failed: {e}")

        asyncio.run_coroutine_threadsafe(_cal(), self._loop)



    def update_bucket_edges(self, bucket_edges: List[Tuple[float, float]]) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_all_bucket_edges(bucket_edges)

            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_actuator_colour(self, actuator_id: int, colour_name: str) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_actuator_color(actuator_id, colour_name)

            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_actuator_freq(self, actuator_id: int, freq_index: int) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_bucket_freq(actuator_id, freq_index)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_actuator_gain(self, actuator_id: int, gain: float) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_actuator_gain(actuator_id, gain)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def update_actuator_activation(self, actuator_id: int, min_duty: int, max_duty: int) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_actuator_activation_range(actuator_id, min_duty, max_duty)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    # ---- Worker main loop ----

    def update_actuator_mapping(self, actuator_id: int, channel: int, address: int) -> None:
        if not (self._loop and self.controller):
            return

        def _do():
            try:
                self.controller.set_actuator_channel_address(int(actuator_id), int(channel), int(address))
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_do)

    def run(self) -> None:
        self._running = True
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._amain())
        finally:
            try:
                loop.close()
            except Exception:
                pass
            self.finished.emit()

    async def _amain(self) -> None:
        """Connect BLE, open mic, run realtime while _running is True."""
        # Build controller
        self.controller = VoiceToHapticsController(
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            chunk_size=1024,
            n_actuators=N_ACTUATORS,
            fmin=DEFAULT_FMIN,
            fmax=DEFAULT_FMAX,
            canonical_K=DEFAULT_K,
            bucket_method="voice_landmark",
            freq_index=self.cfg.global_freq_index,
            noise_gate_db=self.cfg.noise_gate_db,
        )

        mode = self.cfg.mode
        if mode == "colour":
            mode = "color"
        self.controller.set_mode(mode)

        # Apply spectral span + dominance defaults
        try:
            self.controller.set_spectral_span_db(float(self.cfg.spectral_span_db))
        except Exception:
            pass
        try:
            self.controller.set_colour_dominance(bool(self.cfg.dominance_enabled), int(self.cfg.dominance_top_k), int(self.cfg.dominance_group_size), float(self.cfg.dominance_non_dom_gain))
        except Exception:
            pass

        # Apply initial physical mapping (Phase 1)
        try:
            self.controller.set_all_actuator_channel_addresses(self.cfg.actuator_channels, self.cfg.actuator_addresses)
        except Exception:
            pass

        # Apply initial normal-mode bucket edges
        if self.cfg.bucket_edges:
            self.controller.set_all_bucket_edges(self.cfg.bucket_edges)

        # Apply initial colour assignments
        for a, name in enumerate(self.cfg.actuator_colour_names):
            self.controller.set_actuator_color(a, name)

        # Per-actuator settings
        for a in range(N_ACTUATORS):
            self.controller.set_bucket_freq(a, int(self.cfg.actuator_freq_indices[a]))
            self.controller.set_actuator_gain(a, float(self.cfg.actuator_gains[a]))
            self.controller.set_actuator_activation_range(
                a,
                int(self.cfg.actuator_min_duty[a]),
                int(self.cfg.actuator_max_duty[a]),
            )

        self.status.emit("BLE mode: " + str(getattr(self.cfg, "ble_mode", "required")))
        ble_mode = str(getattr(self.cfg, "ble_mode", "required")).strip().lower()
        ble_connected = False

        if ble_mode in ("disabled", "off", "false", "0", "no", "noble", "visual", "visual-only", "visual_only"):
            self.status.emit("BLE disabled -> running visual-only.")
        else:
            self.status.emit("Scanning/connecting Vibraforge BLE…")
            ok = await self.controller.connect_vibraforge()
            if not ok:
                scan = getattr(self.controller, "last_ble_scan", []) or []
                err = getattr(self.controller, "last_ble_error", None)
                if scan:
                    pretty = ", ".join([f"{(n or '?')} [{a}]" for (n, a) in scan[:12]])
                    self.status.emit("BLE device not found. Nearby devices: " + pretty)
                else:
                    self.status.emit("BLE device not found. (No devices discovered.)")
                if err:
                    self.status.emit("BLE debug: " + str(err))

                # Optional fallback: keep running without BLE if requested
                if ble_mode in ("optional", "fallback", "try"):
                    self.status.emit("BLE not connected -> continuing in visual-only mode.")
                else:
                    return
            else:
                ble_connected = True
                name = getattr(self.controller, "ble_connected_name", None)
                addr = getattr(self.controller, "ble_connected_address", None)
                if name or addr:
                    self.status.emit(f"BLE connected: {(name or '?')} [{(addr or '?')}]")

                verify_ok = getattr(self.controller, "ble_verify_ok", None)
                if verify_ok is False:
                    self.status.emit("Connected (warning: could not verify GATT characteristic; proceeding anyway).")

        if ble_connected:
            self.status.emit("Connected. Opening audio source…")
        else:
            self.status.emit("Opening audio source (visual-only)…")
        self.controller.open_audio_source(
            self.cfg.audio_mode,
            input_device_query=self.cfg.input_device_query,
            record_path=self.cfg.record_path,
            playback_path=self.cfg.playback_path,
            playback_to_speaker=self.cfg.playback_to_speaker,
        )
        # Initial colour noise subtraction settings
        self.controller.set_colour_noise_subtract(self.cfg.colour_noise_subtract_enabled, self.cfg.colour_noise_subtract_alpha)
        if self.cfg.debug_bars_enabled:
            try:
                self.debug_meta.emit(self.controller.get_debug_meta())
            except Exception:
                pass
        self.status.emit("Running. ESC = PANIC STOP; Stop button = shutdown.")
        mic_was_ok = True
        try:
            while self._running:
                loop_t0 = time.monotonic()
                intensity = self.controller.process_audio_frame()

                if intensity is None:
                    if mic_was_ok:
                        mic_was_ok = False
                        # Stop haptics immediately if audio broke / stream closed
                        try:
                            await self.controller.panic_stop()
                        except Exception:
                            pass

                        mode_now = str(getattr(self.controller, "audio_mode", getattr(self.cfg, "audio_mode", "live"))).lower()
                        if mode_now in ("replay", "playback"):
                            self.status.emit("Replay finished -> haptics stopped.")
                            self._running = False
                            break
                        else:
                            self.status.emit("Audio source ended/error -> haptics stopped.")
                    await asyncio.sleep(0.05)
                    continue
                else:
                    if not mic_was_ok:
                        mic_was_ok = True
                        self.status.emit("Audio OK -> resuming.")

                # Update GUI
                self.intensity_updated.emit(intensity.copy())

                # Drive haptics
                await self.controller.send_haptic_command(intensity)

                # RMS stats
                mean5, peak5, mean10, peak10 = self.controller.get_rms_stats()
                self.rms_updated.emit(mean5, peak5, mean10, peak10)

                # Debug bars (throttled)
                if self._debug_enabled:
                    now = time.monotonic()
                    if now - self._last_debug_emit_t >= (1.0 / max(1e-6, self._debug_fps)):
                        try:
                            e, c_raw, c_clean = self.controller.compute_debug_features()
                            self.features_updated.emit(e.copy(), c_raw.copy(), c_clean.copy(), float(self.controller.last_rms_db))
                        except Exception:
                            pass
                        self._last_debug_emit_t = now

                # Real-time pacing (important for replay mode)
                mode_now = str(getattr(self.controller, "audio_mode", getattr(self.cfg, "audio_mode", "live"))).lower()
                if mode_now in ("replay", "playback"):
                    target_dt = float(self.controller.chunk_size) / float(self.controller.sr)
                    loop_elapsed = time.monotonic() - loop_t0
                    await asyncio.sleep(max(0.0, target_dt - loop_elapsed))
                else:
                    await asyncio.sleep(0.001)

        except Exception as e:
            self.status.emit(f"Error: {e}")

        finally:
            # The shutdown method is designed specifically to fix the
            # “can’t restart after stop” issue by releasing BLE+audio resources.
            self.status.emit("Stopping (shutdown)…")
            try:
                if self.controller:
                    await self.controller.shutdown()
            except Exception:
                pass
            self.status.emit("Stopped.")


# -----------------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------------


class VoiceHapticsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibraforge – Voice → Haptics")

        self.thread: Optional[QThread] = None
        self.worker: Optional[HapticsWorker] = None

        # ----------------
        # GUI state (used when starting the worker)
        # ----------------
        self.current_mode: str = "normal"  # "normal" or "colour"
        self.current_noise_gate_db: float = -60.0
        self.current_spectral_span_db: float = 25.0
        self.current_dominance_enabled: bool = False
        self.current_dominance_top_k: int = 2
        self.current_dominance_group_size: int = 1
        self.current_dominance_non_dom_gain: float = 0.35
        self.current_global_freq_index: int = 5
        self.current_input_device_query: Optional[str] = None


        # Current per-actuator configuration
        self.actuator_freq_indices = [self.current_global_freq_index] * N_ACTUATORS
        self.actuator_gains = [1.0] * N_ACTUATORS
        self.actuator_min_duty = [0] * N_ACTUATORS
        self.actuator_max_duty = [15] * N_ACTUATORS

        # Phase 1 mapping: logical actuator i -> (channel, address) -> physical id
        self.actuator_channels = [0] * N_ACTUATORS
        self.actuator_addresses = list(range(N_ACTUATORS))


        # Normal-mode bucket edges (Hz). Initialise from voice_landmark template.
        self.bucket_edges = self._template_edges("voice_landmark")

        # Colour palette and assignments
        self.colour_channels = default_voice_color_channels(fmin=DEFAULT_FMIN, fmax=DEFAULT_FMAX)
        self.colour_names = [c.name for c in self.colour_channels]

        # Default assignment: cycle through the first N_ACTUATORS colours.
        self.actuator_colour_names = [self.colour_names[i % len(self.colour_names)] for i in range(N_ACTUATORS)]

        # Presets (still useful: affects freq/gains/noise gate, independent of mode)
        self.presets = {
            "Baseline (flat)": {
                "global_freq": 5,
                "noise_gate_db": -60.0,
                "freqs": [5] * N_ACTUATORS,
                "gains": [1.0] * N_ACTUATORS,
            },
            "Bass heavy (voice body)": {
                "global_freq": 4,
                "noise_gate_db": -55.0,
                "freqs": [3, 3, 4, 5, 5, 6][:N_ACTUATORS],
                "gains": [2.0, 1.8, 1.2, 0.8, 0.5, 0.3][:N_ACTUATORS],
            },
            "Mid presence focus": {
                "global_freq": 5,
                "noise_gate_db": -50.0,
                "freqs": [4, 4, 5, 5, 5, 6][:N_ACTUATORS],
                "gains": [0.8, 1.0, 1.4, 1.4, 1.0, 0.6][:N_ACTUATORS],
            },
        }

        # ----------------
        # Build UI
        # ----------------
        w = QWidget()
        self.setCentralWidget(w)

        # Tabs: keep main UI clean, put heavy debug in its own tab when requested
        outer = QVBoxLayout(w)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.on_tab_close_requested)
        outer.addWidget(self.tabs)

        # --- Main tab with overall scroll (so the UI can go right-first + still be navigable vertically) ---
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_tab_layout.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        self.tabs.addTab(main_tab, "Main")

        # Debug tab is created lazily via the button (Phase 1 fast path)
        self.debug_tab_index = None
        self.open_debug_btn = None
        self.debug_enable_chk = None
        self.debug_note = None
        self.colour_debug_bars = []
        self.canon_debug_bars = []

        # --- Small UI scaling when there are many actuators ---
        if N_ACTUATORS <= 8:
            font_bar = QFont("Consolas", 10)
            font_info = QFont("Consolas", 8)
            act_label_w = 70
            color_w = 56
        elif N_ACTUATORS <= 12:
            font_bar = QFont("Consolas", 9)
            font_info = QFont("Consolas", 8)
            act_label_w = 60
            color_w = 48
        else:
            font_bar = QFont("Consolas", 8)
            font_info = QFont("Consolas", 7)
            act_label_w = 55
            color_w = 44

        # Status
        self.status_label = QLabel("Idle.")
        self.status_label.setFont(QFont("Arial", 11))
        root.addWidget(self.status_label)

        # ========== TOP ROW: presets/help + RMS (moved to top as requested) ==========
        top_row = QHBoxLayout()

        preset_box = QGroupBox("Presets & help")
        pl = QHBoxLayout(preset_box)
        pl.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for k in self.presets.keys():
            self.preset_combo.addItem(k)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        pl.addWidget(self.preset_combo)
        self.help_btn = QPushButton("Help / What does this do?")
        self.help_btn.clicked.connect(self.show_help)
        pl.addStretch(1)
        pl.addWidget(self.help_btn)
        top_row.addWidget(preset_box, 2)

        rms_group = QGroupBox("Audio loudness (RMS dB)")
        rl = QVBoxLayout(rms_group)
        self.rms_label_5s = QLabel("5s mean: --  peak: --")
        self.rms_label_10s = QLabel("10s mean: --  peak: --")
        self.rms_label_5s.setFont(font_bar)
        self.rms_label_10s.setFont(font_bar)
        rl.addWidget(self.rms_label_5s)
        rl.addWidget(self.rms_label_10s)
        top_row.addWidget(rms_group, 1)

        root.addLayout(top_row)

               # ========== CONTROL STRIP (rows-based layout) ==========
        control_strip = QGroupBox("Session / Global controls")
        cs = QGridLayout(control_strip)
        cs.setHorizontalSpacing(10)
        cs.setVerticalSpacing(6)
        cs.setContentsMargins(10, 10, 10, 10)

        # --- Row 0: Start/Stop + Mode + Debug button ---
        row0 = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        row0.addWidget(self.start_btn)
        row0.addWidget(self.stop_btn)

        # Visual-only / No-BLE mode (useful for demos without hardware)
        self.no_ble_chk = QCheckBox("No BLE (visual-only)")
        self.no_ble_chk.setToolTip("Run analysis + visual haptics without any BLE hardware. No BLE connection attempt.")
        row0.addWidget(self.no_ble_chk)

        row0.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Normal (spectral buckets)", userData="normal")
        self.mode_combo.addItem("Colour (timbre channels)", userData="colour")
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        row0.addWidget(self.mode_combo)
        
        self.open_debug_btn = QPushButton("Open Debug tab")
        self.open_debug_btn.clicked.connect(self.open_debug_tab)
        row0.addWidget(self.open_debug_btn)
        row0.addStretch(1)
        cs.addLayout(row0, 0, 0, 1, 5)

        # --- Row 1: Audio input ---
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Audio input:"))
        self.input_combo = QComboBox()
        self.input_combo.addItem("Default (Windows)", userData=None)
        self.input_combo.addItem("Raw headset mic (Realtek)", userData="Headset Microphone")
        self.input_combo.addItem("OBS RNNoise (VB-CABLE)", userData="CABLE Output")
        self.input_combo.currentIndexChanged.connect(self.on_input_device_changed)
        row1.addWidget(self.input_combo)
        row1.addStretch(1)
        self.current_input_device_query = self.input_combo.currentData()
        cs.addLayout(row1, 1, 0, 1, 5)

        # --- Row 2: Audio mode + Calibrate ---
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Audio mode:"))
        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItem("Live (microphone)", userData="live")
        self.audio_mode_combo.addItem("Record (mic → WAV + haptics)", userData="record")
        self.audio_mode_combo.addItem("Replay (WAV → speaker + haptics)", userData="replay")
        self.audio_mode_combo.setCurrentIndex(0)
        self.audio_mode_combo.currentIndexChanged.connect(self.on_audio_mode_changed)
        row2.addWidget(self.audio_mode_combo)
        
        self.calibrate_btn = QPushButton("Calibrate room noise (2s)")
        self.calibrate_btn.clicked.connect(self.on_calibrate_clicked)
        row2.addWidget(self.calibrate_btn)
        row2.addStretch(1)
        cs.addLayout(row2, 2, 0, 1, 5)

        # --- Row 3: Record path ---
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Record path:"))
        self.record_path_edit = QLineEdit("recordings/voice_recording.wav")
        self.record_browse_btn = QPushButton("Browse…")
        self.record_browse_btn.clicked.connect(self.browse_record_path)
        row3.addWidget(self.record_path_edit)
        row3.addWidget(self.record_browse_btn)
        row3.addStretch(1)
        cs.addLayout(row3, 3, 0, 1, 5)

        # --- Row 4: Replay path + speaker ---
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Replay file:"))
        self.playback_path_edit = QLineEdit("")
        self.playback_browse_btn = QPushButton("Browse…")
        self.playback_browse_btn.clicked.connect(self.browse_playback_path)
        row4.addWidget(self.playback_path_edit)
        row4.addWidget(self.playback_browse_btn)
        
        self.play_to_speaker_chk = QCheckBox("Play to speaker")
        self.play_to_speaker_chk.setChecked(True)
        row4.addWidget(self.play_to_speaker_chk)
        row4.addStretch(1)
        cs.addLayout(row4, 4, 0, 1, 5)

        # --- Row 5: Global frequency + Noise gate ---
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Global vibration freq index (0–7):"))
        self.global_freq_combo = QComboBox()
        for i in range(8):
            self.global_freq_combo.addItem(str(i))
        self.global_freq_combo.setCurrentIndex(self.current_global_freq_index)
        self.global_freq_combo.currentIndexChanged.connect(self.on_global_freq_changed)
        row5.addWidget(self.global_freq_combo)
        
        self.noise_label = QLabel(f"Noise gate (dB): {self.current_noise_gate_db:.1f}")
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(-80.0, -20.0)
        self.noise_spin.setSingleStep(1.0)
        self.noise_spin.setValue(self.current_noise_gate_db)
        self.noise_spin.valueChanged.connect(self.on_noise_gate_changed)
        row5.addWidget(self.noise_label)
        row5.addWidget(self.noise_spin)
        row5.addStretch(1)
        cs.addLayout(row5, 5, 0, 1, 5)

        # --- Row 6: Spectral span ---
        row6 = QHBoxLayout()
        self.spectral_span_label = QLabel(f"Spectral span (dB): {self.current_spectral_span_db:.0f}")
        self.spectral_span_slider = QSlider(Qt.Horizontal)
        self.spectral_span_slider.setRange(5, 60)
        self.spectral_span_slider.setSingleStep(1)
        self.spectral_span_slider.setValue(int(self.current_spectral_span_db))
        self.spectral_span_slider.valueChanged.connect(self.on_spectral_span_changed)
        row6.addWidget(self.spectral_span_label)
        row6.addWidget(self.spectral_span_slider)
        row6.addStretch(1)
        cs.addLayout(row6, 6, 0, 1, 5)

        # --- Row 7: Dominance (grouped) ---
        row7 = QHBoxLayout()
        self.dominance_chk = QCheckBox("Dominance (grouped)")
        self.dominance_chk.setChecked(bool(self.current_dominance_enabled))
        self.dominance_chk.stateChanged.connect(self.on_dominance_changed)

        self.dominance_k_spin = QSpinBox()
        self.dominance_k_spin.setRange(1, min(3, max(1, len(self.colour_names))))
        self.dominance_k_spin.setValue(int(self.current_dominance_top_k))
        self.dominance_k_spin.valueChanged.connect(self.on_dominance_changed)

        self.dominance_group_spin = QSpinBox()
        self.dominance_group_spin.setRange(1, 3)
        self.dominance_group_spin.setValue(int(getattr(self, "current_dominance_group_size", 1)))
        self.dominance_group_spin.valueChanged.connect(self.on_dominance_changed)

        self.dominance_other_gain_spin = QDoubleSpinBox()
        self.dominance_other_gain_spin.setRange(0.0, 1.0)
        self.dominance_other_gain_spin.setSingleStep(0.05)
        self.dominance_other_gain_spin.setDecimals(2)
        self.dominance_other_gain_spin.setValue(float(getattr(self, "current_dominance_non_dom_gain", 0.35)))
        self.dominance_other_gain_spin.valueChanged.connect(self.on_dominance_changed)

        row7.addWidget(self.dominance_chk)
        row7.addWidget(QLabel("Top-K:"))
        row7.addWidget(self.dominance_k_spin)
        row7.addSpacing(10)
        row7.addWidget(QLabel("Group size:"))
        row7.addWidget(self.dominance_group_spin)
        row7.addSpacing(10)
        row7.addWidget(QLabel("Others gain:"))
        row7.addWidget(self.dominance_other_gain_spin)
        row7.addStretch(1)
        cs.addLayout(row7, 7, 0, 1, 5)

        # --- Row 8: Per-colour noise subtraction ---
        row8 = QHBoxLayout()
        self.colour_subtract_chk = QCheckBox("Per-colour noise subtraction")
        self.colour_subtract_chk.setChecked(True)
        self.colour_subtract_chk.stateChanged.connect(self.on_colour_subtract_changed)
        self.colour_subtract_alpha_spin = QDoubleSpinBox()
        self.colour_subtract_alpha_spin.setRange(0.0, 3.0)
        self.colour_subtract_alpha_spin.setSingleStep(0.1)
        self.colour_subtract_alpha_spin.setValue(1.0)
        self.colour_subtract_alpha_spin.valueChanged.connect(self.on_colour_subtract_alpha_changed)
        row8.addWidget(self.colour_subtract_chk)
        row8.addWidget(self.colour_subtract_alpha_spin)
        row8.addStretch(1)
        cs.addLayout(row8, 8, 0, 1, 5)

        # Apply initial enabling/disabling based on mode
        self.on_audio_mode_changed(self.audio_mode_combo.currentIndex())

        root.addWidget(control_strip)

        # ========== DEBUG BARS (canonical + colours) ==========

        # ESC = panic stop
        self._esc = QShortcut(QKeySequence("Esc"), self)
        self._esc.activated.connect(self.panic_stop)

        # ========== LIVE ACTUATORS: intensity left + controls to the right (per-row) ==========
        live_box = QGroupBox("Live actuators")
        live_layout = QVBoxLayout(live_box)
        live_layout.setContentsMargins(10, 10, 10, 10)

        # Normal-mode template row (shown only in normal mode)
        self.template_bar = QWidget()
        template_row = QHBoxLayout(self.template_bar)
        template_row.setContentsMargins(0, 0, 0, 0)
        template_row.addWidget(QLabel("Bucket template:"))
        self.bucket_template_combo = QComboBox()
        self.bucket_template_combo.addItem("Voice-landmark (default)", userData="voice_landmark")
        self.bucket_template_combo.addItem("ERB-spaced", userData="erb")
        self.bucket_template_combo.addItem("Mel-spaced", userData="mel")
        self.bucket_template_combo.addItem("Log-spaced", userData="log")
        self.bucket_template_combo.addItem("Linear-spaced", userData="linear")
        self.bucket_template_combo.currentIndexChanged.connect(self.on_bucket_template_changed)
        template_row.addWidget(self.bucket_template_combo)
        self.apply_buckets_btn = QPushButton("Apply bucket edges")
        self.apply_buckets_btn.clicked.connect(self.apply_bucket_edges)
        template_row.addStretch(1)
        template_row.addWidget(self.apply_buckets_btn)
        live_layout.addWidget(self.template_bar)

        hint = QLabel("Tip: you can overlap buckets for experiments (a canonical band can contribute to multiple buckets).")
        hint.setFont(QFont("Arial", 9))
        hint.setStyleSheet("color: #666666;")
        live_layout.addWidget(hint)

        # Scroll area for actuator rows (kicks in automatically if there are many)
        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        live_layout.addWidget(rows_scroll)

        rows_widget = QWidget()
        rows_scroll.setWidget(rows_widget)
        grid = QGridLayout(rows_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # Column headers
        hdr_font = QFont("Arial", 9)
        hdr_font.setBold(True)
        headers = [
            "Act",
            "Ch",
            "Addr",
            "Phys",
            "Intensity",
            "Bar",
            "Freq",
            "Gain",
            "Min",
            "Max",
            "Bucket lo",
            "Bucket hi",
            "Colour",
            "Info",
        ]
        self._hdr_widgets: List[QLabel] = []
        for c, txt in enumerate(headers):
            h = QLabel(txt)
            h.setFont(hdr_font)
            h.setStyleSheet("color: #444444;")
            grid.addWidget(h, 0, c)
            self._hdr_widgets.append(h)

        # Per-actuator widgets (rows)
        self.intensity_labels = []
        self.intensity_color_boxes = []
        self.info_labels = []
        self.freq_combos = []
        self.gain_spins = []
        self.min_duty_spins = []
        self.max_duty_spins = []
        self.bucket_edge_spins = []
        self.colour_combos = []
        self.channel_spins = []
        self.address_spins = []
        self.phys_labels = []
        self.act_labels = []

        for i in range(N_ACTUATORS):
            r = i + 1

            act_lbl = QLabel(f"{i} (Ch{self.actuator_channels[i]}:{self.actuator_addresses[i]})")
            act_lbl.setMinimumWidth(act_label_w)
            grid.addWidget(act_lbl, r, 0)

            self.act_labels.append(act_lbl)

            ch_spin = QSpinBox()
            ch_spin.setRange(0, MAX_CHANNELS - 1)
            ch_spin.setValue(int(self.actuator_channels[i]))
            ch_spin.valueChanged.connect(lambda _v, a=i: self.on_actuator_mapping_changed(a))
            self.channel_spins.append(ch_spin)
            grid.addWidget(ch_spin, r, 1)

            addr_spin = QSpinBox()
            addr_spin.setRange(0, ADDRS_PER_CHANNEL - 1)
            addr_spin.setValue(int(self.actuator_addresses[i]))
            addr_spin.valueChanged.connect(lambda _v, a=i: self.on_actuator_mapping_changed(a))
            self.address_spins.append(addr_spin)
            grid.addWidget(addr_spin, r, 2)

            phys = int(self.actuator_channels[i]) * ADDRS_PER_CHANNEL + int(self.actuator_addresses[i])
            phys_lbl = QLabel(str(phys))
            phys_lbl.setMinimumWidth(45)
            self.phys_labels.append(phys_lbl)
            grid.addWidget(phys_lbl, r, 3)


            # Color intensity box
            color_box = QWidget()
            color_box.setMinimumHeight(24)
            color_box.setMinimumWidth(color_w)
            color_box.setStyleSheet(f"background-color: {INTENSITY_PALETTE[0]}; border-radius: 4px; border: 1px solid #333333;")
            self.intensity_color_boxes.append(color_box)
            grid.addWidget(color_box, r, 4)

            # Intensity bar + text
            main = QLabel(f"[{bar(0)}] 0/15")
            main.setFont(font_bar)
            self.intensity_labels.append(main)
            grid.addWidget(main, r, 5)

            # Per-actuator freq
            combo = QComboBox()
            for k in range(8):
                combo.addItem(str(k))
            combo.setCurrentIndex(self.actuator_freq_indices[i])
            combo.currentIndexChanged.connect(lambda idx, a=i: self.on_actuator_freq_changed(a, idx))
            self.freq_combos.append(combo)
            grid.addWidget(combo, r, 6)

            # Per-actuator gain
            gain_spin = QDoubleSpinBox()
            gain_spin.setRange(0.0, 3.0)
            gain_spin.setSingleStep(0.1)
            gain_spin.setValue(self.actuator_gains[i])
            gain_spin.valueChanged.connect(lambda val, a=i: self.on_actuator_gain_changed(a, val))
            self.gain_spins.append(gain_spin)
            grid.addWidget(gain_spin, r, 7)

            # Activation range (always visible; useful in both modes)
            mn = QSpinBox()
            mn.setRange(0, 15)
            mn.setValue(self.actuator_min_duty[i])
            mn.valueChanged.connect(lambda _v, a=i: self.on_actuator_activation_changed(a))
            self.min_duty_spins.append(mn)
            grid.addWidget(mn, r, 8)

            mx = QSpinBox()
            mx.setRange(0, 15)
            mx.setValue(self.actuator_max_duty[i])
            mx.valueChanged.connect(lambda _v, a=i: self.on_actuator_activation_changed(a))
            self.max_duty_spins.append(mx)
            grid.addWidget(mx, r, 9)

            # Normal-mode bucket edges (lo/hi)
            lo = QDoubleSpinBox()
            hi = QDoubleSpinBox()
            for sp in (lo, hi):
                sp.setRange(0.0, 20000.0)
                sp.setSingleStep(10.0)
                sp.setDecimals(0)
            lo.setValue(self.bucket_edges[i][0])
            hi.setValue(self.bucket_edges[i][1])
            lo.valueChanged.connect(lambda _v, a=i: self._on_bucket_spin_changed(a))
            hi.valueChanged.connect(lambda _v, a=i: self._on_bucket_spin_changed(a))
            self.bucket_edge_spins.append((lo, hi))
            grid.addWidget(lo, r, 10)
            grid.addWidget(hi, r, 11)

            # Colour-mode combo
            ccombo = QComboBox()
            for name in self.colour_names:
                ccombo.addItem(name)
            ccombo.setCurrentText(self.actuator_colour_names[i])
            ccombo.currentIndexChanged.connect(lambda _idx, a=i: self.on_actuator_colour_changed(a))
            self.colour_combos.append(ccombo)
            grid.addWidget(ccombo, r, 12)

            # Info label (kept on the same row to avoid “drifting distance”)
            info = QLabel("–")
            info.setFont(font_info)
            info.setStyleSheet("color: #888888;")
            info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.info_labels.append(info)
            grid.addWidget(info, r, 13)

        root.addWidget(live_box)

        # Start with normal mode visible
        self._refresh_mode_visibility()
        self._refresh_info_labels()

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------
    def on_input_device_changed(self, idx: int) -> None:
        self.current_input_device_query = self.input_combo.currentData()
        self.status_label.setText("Audio input set. Stop/Start to apply.")

    def _debug_enabled(self) -> bool:
        return bool(self.debug_enable_chk.isChecked()) if self.debug_enable_chk else False


    def on_audio_mode_changed(self, idx: int) -> None:
        self.current_audio_mode = str(self.audio_mode_combo.currentData() or "live")
        # Enable/disable path fields based on mode
        is_record = self.current_audio_mode == "record"
        is_replay = self.current_audio_mode == "replay"

        self.record_path_edit.setEnabled(is_record)
        self.record_browse_btn.setEnabled(is_record)

        self.playback_path_edit.setEnabled(is_replay)
        self.playback_browse_btn.setEnabled(is_replay)
        self.play_to_speaker_chk.setEnabled(is_replay)

        # If running, switch live (no restart)
        if self.worker:
            self.worker.update_audio_source(
                self.current_audio_mode,
                self.current_input_device_query,
                self.record_path_edit.text().strip() or None,
                self.playback_path_edit.text().strip() or None,
                bool(self.play_to_speaker_chk.isChecked()),
            )

    def browse_record_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select WAV file to record", self.record_path_edit.text(), "WAV files (*.wav)")
        if path:
            self.record_path_edit.setText(path)

    def browse_playback_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select WAV file to replay", self.playback_path_edit.text(), "WAV files (*.wav)")
        if path:
            self.playback_path_edit.setText(path)

    def on_calibrate_clicked(self) -> None:
        if not self.worker:
            QMessageBox.information(self, "Calibration", "Start the session first, then press Calibrate.\nCalibration runs in the worker thread.")
            return
        self.worker.request_calibration(duration_s=2.0, percentile=50.0, gate_margin_db=3.0, colour_percentile=50.0)

    def on_colour_subtract_changed(self, _state: int) -> None:
        if self.worker:
            self.worker.update_colour_noise_subtract(
                bool(self.colour_subtract_chk.isChecked()),
                float(self.colour_subtract_alpha_spin.value()),
            )

    def on_colour_subtract_alpha_changed(self, _val: float) -> None:
        if self.worker:
            self.worker.update_colour_noise_subtract(
                bool(self.colour_subtract_chk.isChecked()),
                float(self.colour_subtract_alpha_spin.value()),
            )

    def on_debug_toggled(self, _state: int) -> None:
        enabled = bool(self.debug_enable_chk.isChecked()) if self.debug_enable_chk is not None else False
        # If running, just toggle worker emission
        if self.worker:
            self.worker.update_debug_enabled(enabled)


    # ---------------- Debug tab (lazy) ----------------
    def open_debug_tab(self) -> None:
        # Create the Debug tab only when requested, so it doesn't squeeze the main layout.
        if self.debug_tab_index is None:
            debug_tab = QWidget()
            debug_layout = QVBoxLayout(debug_tab)

            debug_scroll = QScrollArea()
            debug_scroll.setWidgetResizable(True)
            debug_layout.addWidget(debug_scroll)

            debug_content = QWidget()
            debug_scroll.setWidget(debug_content)
            dbg_root = QVBoxLayout(debug_content)
            dbg_root.setContentsMargins(10, 10, 10, 10)

            self._build_debug_tab(dbg_root)

            self.debug_tab_index = self.tabs.addTab(debug_tab, "Debug")
            self.tabs.setCurrentIndex(self.debug_tab_index)

            if self.open_debug_btn is not None:
                self.open_debug_btn.setEnabled(False)
        else:
            self.tabs.setCurrentIndex(self.debug_tab_index)

    def on_tab_close_requested(self, index: int) -> None:
        # Keep "Main" tab always. Allow closing the Debug tab.
        if index == 0:
            return

        if self.debug_tab_index is not None and index == self.debug_tab_index:
            # Disable debug emission at the worker side
            if self.worker:
                self.worker.update_debug_enabled(False)

            # Remove tab
            self.tabs.removeTab(index)
            self.debug_tab_index = None

            # Re-enable open button
            if self.open_debug_btn is not None:
                self.open_debug_btn.setEnabled(True)

            # Drop widget refs (they were deleted with the tab)
            self.debug_enable_chk = None
            self.debug_note = None
            self.colour_debug_bars = []
            self.canon_debug_bars = []

    def _build_debug_tab(self, parent_layout: QVBoxLayout) -> None:
        # Debug controls
        header = QHBoxLayout()
        self.debug_enable_chk = QCheckBox("Enable debug bargraphs (20 fps)")
        self.debug_enable_chk.setChecked(False)
        self.debug_enable_chk.stateChanged.connect(self.on_debug_toggled)
        header.addWidget(self.debug_enable_chk)

        # Which colour energies should the debug bars display?
        # Raw = pre noise-floor subtraction; Calibrated = after subtraction (current calibration).
        header.addWidget(QLabel("Colour view:"))
        self.debug_colour_view_combo = QComboBox()
        self.debug_colour_view_combo.addItems(["Raw", "Calibrated"])
        self.debug_colour_view_combo.setCurrentIndex(1)
        header.addWidget(self.debug_colour_view_combo)

        self.debug_note = QLabel("Tip: enable to visualize spectral behaviour + noise floor effects.")
        header.addWidget(self.debug_note)
        header.addStretch(1)
        parent_layout.addLayout(header)

        # --- Colours (collapsible) ---
        colours_box = QGroupBox("Colour energies")
        colours_box.setCheckable(True)
        colours_box.setChecked(True)
        cb = QVBoxLayout(colours_box)

        colours_inner = QWidget()
        cg = QGridLayout(colours_inner)
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(4)

        self.colour_debug_bars = []
        for i, name in enumerate(self.colour_names):
            lab = QLabel(name)
            pb = QProgressBar()
            pb.setRange(0, 100)
            pb.setValue(0)
            pb.setTextVisible(False)
            self.colour_debug_bars.append((lab, pb))
            cg.addWidget(lab, i, 0)
            cg.addWidget(pb, i, 1)

        cb.addWidget(colours_inner)

        # True collapse behaviour
        colours_box.toggled.connect(colours_inner.setVisible)
        parent_layout.addWidget(colours_box)

        # --- Canonical bands (collapsible + scroll) ---
        canon_box = QGroupBox("Canonical bands")
        canon_box.setCheckable(True)
        canon_box.setChecked(True)
        canon_v = QVBoxLayout(canon_box)

        canon_scroll = QScrollArea()
        canon_scroll.setWidgetResizable(True)
        canon_inner = QWidget()
        canon_scroll.setWidget(canon_inner)
        canon_grid = QGridLayout(canon_inner)
        canon_grid.setHorizontalSpacing(8)
        canon_grid.setVerticalSpacing(4)

        self.canon_debug_bars = []
        for k in range(DEFAULT_K):
            lab = QLabel(f"Band {k}")
            pb = QProgressBar()
            pb.setRange(0, 100)
            pb.setValue(0)
            pb.setTextVisible(False)
            self.canon_debug_bars.append((lab, pb))
            canon_grid.addWidget(lab, k, 0)
            canon_grid.addWidget(pb, k, 1)

        canon_v.addWidget(canon_scroll)

        canon_box.toggled.connect(canon_scroll.setVisible)
        parent_layout.addWidget(canon_box)

        parent_layout.addStretch(1)

    def on_debug_meta(self, meta: object) -> None:
        # meta: {"canonical_centers_hz": [...], "colour_names": [...]}
        try:
            d = dict(meta)  # type: ignore
        except Exception:
            return
        centers = d.get("canonical_centers_hz")
        if centers:
            # Update canonical labels with frequency tooltips
            for k, (lab, _pb) in enumerate(self.canon_debug_bars):
                if k < len(centers):
                    lab.setToolTip(f"{float(centers[k]):.0f} Hz")

    def _energies_to_pct(self, arr) -> list[int]:
        import numpy as _np
        a = _np.asarray(arr, dtype=_np.float32).reshape(-1)
        if a.size == 0:
            return []
        db = 10.0 * _np.log10(a + 1e-12)
        if not _np.isfinite(db).any():
            return [0] * int(a.size)
        mx = float(_np.max(db[_np.isfinite(db)]))
        rel = db - mx
        rel = _np.clip(rel, -40.0, 0.0)
        pct = ((rel + 40.0) / 40.0) * 100.0
        return [int(x) for x in pct]

    def on_features_updated(self, canonical: object, colours_raw: object, colours_cal: object, rms_db: float) -> None:
        if not (self.debug_enable_chk and self.debug_enable_chk.isChecked()):
            return
        # Canonical
        canon_pct = self._energies_to_pct(canonical)
        for i, (_lab, pb) in enumerate(self.canon_debug_bars):
            if i < len(canon_pct):
                pb.setValue(canon_pct[i])

        # Colours
        # Choose raw vs calibrated view
        use_raw = bool(getattr(self, 'debug_colour_view_combo', None) and (self.debug_colour_view_combo.currentIndex() == 0))
        colours = colours_raw if use_raw else colours_cal
        col_pct = self._energies_to_pct(colours)
        for i, (_lab, pb) in enumerate(self.colour_debug_bars):
            if i < len(col_pct):
                pb.setValue(col_pct[i])

    def on_calibration_done(self, noise_gate_db: float, noise_floor_colour: object) -> None:
        # Update GUI noise gate control and push to worker/controller
        gate = float(noise_gate_db)
        self.noise_spin.blockSignals(True)
        self.noise_spin.setValue(gate)
        self.noise_spin.blockSignals(False)
        self.current_noise_gate_db = gate
        self.noise_label.setText(f"Noise gate (dB): {gate:.1f}")
        if self.worker:
            self.worker.update_noise_gate(gate)
        self.status_label.setText(f"Calibration applied. noise_gate={gate:.1f} dB")
    def _template_edges(self, bucket_method: str) -> List[Tuple[float, float]]:
        """Generate bucket edges (Hz) using bucket_classification.make_bucket_config."""
        cfg = make_bucket_config(
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            fmin=DEFAULT_FMIN,
            fmax=DEFAULT_FMAX,
            K=DEFAULT_K,
            canonical_scale="erb",
            M=N_ACTUATORS,
            bucket_method=bucket_method,  # voice_landmark / mel / erb / log / linear
            inner_scale="erb",
        )
        return list(cfg.bucket_edges_hz)

    # ------------------------------------------------------------------
    # Start/stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        # Validate audio-mode prerequisites before launching the worker
        audio_mode = str(self.current_audio_mode or "live")
        record_path = self.record_path_edit.text().strip() or None
        playback_path = self.playback_path_edit.text().strip() or None

        if audio_mode == "record" and not record_path:
            QMessageBox.warning(self, "Record mode", "Please choose a WAV file path for recording.")
            return

        if audio_mode in ("replay", "playback"):
            # Convenience: if replay path is empty but the record path exists, reuse it.
            if not playback_path and record_path and os.path.exists(str(record_path)):
                playback_path = record_path
                self.playback_path_edit.setText(str(playback_path))
            if not playback_path:
                QMessageBox.warning(self, "Replay mode", "Please choose a WAV file to replay.")
                return
            if not os.path.exists(str(playback_path)):
                QMessageBox.warning(self, "Replay mode", f"Replay file not found:\n{playback_path}")
                return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        runtime_cfg = RuntimeConfig(
            mode=self.current_mode,
            noise_gate_db=self.current_noise_gate_db,
            spectral_span_db=self.current_spectral_span_db,
            dominance_enabled=bool(self.current_dominance_enabled),
            dominance_top_k=int(self.current_dominance_top_k),
            dominance_group_size=int(self.current_dominance_group_size),
            dominance_non_dom_gain=float(self.current_dominance_non_dom_gain),
            global_freq_index=self.current_global_freq_index,
            bucket_edges=list(self.bucket_edges),
            actuator_colour_names=list(self.actuator_colour_names),
            actuator_freq_indices=list(self.actuator_freq_indices),
            actuator_gains=list(self.actuator_gains),
            actuator_min_duty=list(self.actuator_min_duty),
            actuator_max_duty=list(self.actuator_max_duty),
            actuator_channels=list(self.actuator_channels),
            actuator_addresses=list(self.actuator_addresses),
            input_device_query=self.current_input_device_query,
            audio_mode=audio_mode,
            record_path=record_path,
            playback_path=playback_path,
            playback_to_speaker=bool(self.play_to_speaker_chk.isChecked()),
            ble_mode=("disabled" if (getattr(self, 'no_ble_chk', None) is not None and self.no_ble_chk.isChecked()) else "required"),
            debug_bars_enabled=bool(self.debug_enable_chk.isChecked()) if (self.debug_enable_chk is not None) else False,
            colour_noise_subtract_enabled=bool(self.colour_subtract_chk.isChecked()),
            colour_noise_subtract_alpha=float(self.colour_subtract_alpha_spin.value()),
        )

        self.thread = QThread()
        self.worker = HapticsWorker(runtime_cfg)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.intensity_updated.connect(self.on_intensity)
        self.worker.status.connect(self.status_label.setText)
        self.worker.rms_updated.connect(self.on_rms_updated)
        self.worker.features_updated.connect(self.on_features_updated)
        self.worker.debug_meta.connect(self.on_debug_meta)
        self.worker.calibration_done.connect(self.on_calibration_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_stopped)

        self.thread.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop(immediate_panic=True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping…")

    def panic_stop(self) -> None:
        if self.worker:
            self.worker.panic_stop()
        self.status_label.setText("PANIC STOP sent (ESC).")

    def _on_stopped(self) -> None:
        self.worker = None
        self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Idle (stopped).")

    # ------------------------------------------------------------------
    # GUI updates
    # ------------------------------------------------------------------

    def on_intensity(self, arr) -> None:
        try:
            arr = np.asarray(arr).astype(int)
            for i in range(min(N_ACTUATORS, len(arr))):
                v = int(arr[i])
                v = int(np.clip(v, 0, 15))  # Ensure in range 0-15
                
                # Update text label
                self.intensity_labels[i].setText(f"[{bar(v)}] {v}/15")
                
                # Update color box based on intensity
                color = INTENSITY_PALETTE[v]
                self.intensity_color_boxes[i].setStyleSheet(
                    f"background-color: {color}; border-radius: 4px; border: 1px solid #333333;"
                )
            
            self._refresh_info_labels()
        except Exception as e:
            self.status_label.setText(f"UI update error: {e}")

    def _refresh_info_labels(self) -> None:
        """Refresh the small per-actuator explanation labels."""
        if self.current_mode == "normal":
            for i in range(N_ACTUATORS):
                lo, hi = self.bucket_edges[i]
                self.info_labels[i].setText(
                    f"Band: {fmt_hz(lo, hi)} | active duty: {self.actuator_min_duty[i]}…{self.actuator_max_duty[i]}"
                )
        else:
            # Colour mode: show colour definition ranges
            name_to_ranges = {c.name: c.ranges_hz for c in self.colour_channels}
            for i in range(N_ACTUATORS):
                cname = self.actuator_colour_names[i]
                ranges = name_to_ranges.get(cname, [])
                ranges_txt = ", ".join(fmt_hz(lo, hi) for lo, hi in ranges) if ranges else "(custom)"
                self.info_labels[i].setText(
                    f"Colour: {cname} | ranges: {ranges_txt} | active duty: {self.actuator_min_duty[i]}…{self.actuator_max_duty[i]}"
                )

    def on_rms_updated(self, mean5, peak5, mean10, peak10) -> None:
        def fmt(x):
            if np.isnan(x):
                return "--"
            return f"{x:5.1f} dB"

        self.rms_label_5s.setText(f"5s mean: {fmt(mean5)}   peak: {fmt(peak5)}")
        self.rms_label_10s.setText(f"10s mean: {fmt(mean10)}  peak: {fmt(peak10)}")

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def on_mode_changed(self, idx: int) -> None:
        mode = self.mode_combo.itemData(idx)
        self.current_mode = str(mode)
        self._refresh_mode_visibility()
        self._refresh_info_labels()

        if self.worker:
            self.worker.update_mode(self.current_mode)

    def _refresh_mode_visibility(self) -> None:
        is_normal = self.current_mode == "normal"
        is_colour = self.current_mode == "colour"

        # Top bucket-template bar only matters in normal mode
        if hasattr(self, "template_bar"):
            self.template_bar.setVisible(is_normal)

        # Column visibility: keep a single actuator table, switch relevant columns.
        # Header indices (see `headers` list when building the grid):
        # Bucket lo=10, Bucket hi=11, Colour=12.
        if hasattr(self, "_hdr_widgets") and len(self._hdr_widgets) >= 13:
            self._hdr_widgets[10].setVisible(is_normal)
            self._hdr_widgets[11].setVisible(is_normal)
            self._hdr_widgets[12].setVisible(is_colour)

        for i in range(N_ACTUATORS):
            try:
                lo, hi = self.bucket_edge_spins[i]
                lo.setVisible(is_normal)
                hi.setVisible(is_normal)
            except Exception:
                pass
            try:
                self.colour_combos[i].setVisible(is_colour)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Controls callbacks
    # ------------------------------------------------------------------

    def on_global_freq_changed(self, idx: int) -> None:
        self.current_global_freq_index = int(idx)

        # Update each actuator combo to match
        for i, combo in enumerate(self.freq_combos):
            if combo.currentIndex() != idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
                self.on_actuator_freq_changed(i, idx)

    def on_actuator_freq_changed(self, actuator_id: int, idx: int) -> None:
        self.actuator_freq_indices[actuator_id] = int(idx)
        if self.worker:
            self.worker.update_actuator_freq(actuator_id, int(idx))

    def on_actuator_mapping_changed(self, actuator_id: int) -> None:
        a = int(actuator_id)
        if a < 0 or a >= N_ACTUATORS:
            return
        ch = int(self.channel_spins[a].value())
        ad = int(self.address_spins[a].value())
        self.actuator_channels[a] = ch
        self.actuator_addresses[a] = ad
        phys = ch * ADDRS_PER_CHANNEL + ad
        try:
            self.phys_labels[a].setText(str(phys))
        except Exception:
            pass
        # Also update Act label text for quick visibility
        try:
            self.act_labels[a].setText(f"{a} (Ch{ch}:{ad})")
        except Exception:
            pass
        # If running, push mapping to worker live
        if self.worker:
            self.worker.update_actuator_mapping(a, ch, ad)

    def on_actuator_gain_changed(self, actuator_id: int, value: float) -> None:
        self.actuator_gains[actuator_id] = float(value)
        if self.worker:
            self.worker.update_actuator_gain(actuator_id, float(value))

    def on_noise_gate_changed(self, value: float) -> None:
        self.current_noise_gate_db = float(value)
        self.noise_label.setText(f"Noise gate (dB): {self.current_noise_gate_db:.1f}")
        if self.worker:
            self.worker.update_noise_gate(self.current_noise_gate_db)

    def on_spectral_span_changed(self, value: int) -> None:
        self.current_spectral_span_db = float(value)
        if hasattr(self, 'spectral_span_label') and self.spectral_span_label is not None:
            self.spectral_span_label.setText(f"Spectral span (dB): {self.current_spectral_span_db:.0f}")
        if self.worker:
            self.worker.update_spectral_span_db(self.current_spectral_span_db)

    def on_dominance_changed(self, _value: int | None = None) -> None:
        # Dominance controls share this handler
        enabled = bool(self.dominance_chk.isChecked()) if hasattr(self, 'dominance_chk') else False
        top_k = int(self.dominance_k_spin.value()) if hasattr(self, 'dominance_k_spin') else 2
        group_size = int(self.dominance_group_spin.value()) if hasattr(self, 'dominance_group_spin') else 1
        non_dom_gain = float(self.dominance_other_gain_spin.value()) if hasattr(self, 'dominance_other_gain_spin') else 0.35

        self.current_dominance_enabled = enabled
        self.current_dominance_top_k = top_k
        self.current_dominance_group_size = group_size
        self.current_dominance_non_dom_gain = non_dom_gain

        if self.worker:
            self.worker.update_colour_dominance(enabled, top_k, group_size, non_dom_gain)

    # ------------------------------------------------------------------
    # Normal mode: bucket edges
    # ------------------------------------------------------------------

    def _on_bucket_spin_changed(self, actuator_id: int) -> None:
        """Update local bucket_edges + info labels when the user edits lo/hi.

        We keep the actual apply-to-worker action behind the "Apply bucket edges"
        button (same behavior as before), but we still want the per-row "Info"
        label to remain accurate while editing.
        """
        try:
            lo_sp, hi_sp = self.bucket_edge_spins[actuator_id]
            lo = float(lo_sp.value())
            hi = float(hi_sp.value())
            self.bucket_edges[actuator_id] = (lo, hi)
            self._refresh_info_labels()
        except Exception:
            pass

    def on_bucket_template_changed(self, idx: int) -> None:
        method = self.bucket_template_combo.itemData(idx)
        if not method:
            return
        self.bucket_edges = self._template_edges(str(method))

        # Update spinboxes
        for i, (lo_sp, hi_sp) in enumerate(self.bucket_edge_spins):
            lo_sp.blockSignals(True)
            hi_sp.blockSignals(True)
            lo_sp.setValue(self.bucket_edges[i][0])
            hi_sp.setValue(self.bucket_edges[i][1])
            lo_sp.blockSignals(False)
            hi_sp.blockSignals(False)

        self._refresh_info_labels()

    def apply_bucket_edges(self) -> None:
        """Apply bucket edges with validation and auto-swap."""
        edges: List[Tuple[float, float]] = []
        had_issues = False
        
        for i, (lo_sp, hi_sp) in enumerate(self.bucket_edge_spins):
            lo = float(lo_sp.value())
            hi = float(hi_sp.value())
            
            # ========== FIX #1: Auto-swap if user got them backwards ==========
            if hi < lo:
                lo, hi = hi, lo
                lo_sp.setValue(lo)
                hi_sp.setValue(hi)
                self.status_label.setText(f"Bucket {i}: swapped lo/hi (was {hi:.0f}–{lo:.0f})")
                had_issues = True
            
            # ========== FIX #2: Validate ranges are within allowed limits ==========
            if lo < 0 or hi > 20000:
                self.status_label.setText(f"Bucket {i}: out of range [0–20000 Hz]. Skipping apply.")
                return  # Block apply if out of range
            
            edges.append((lo, hi))
        
        self.bucket_edges = edges
        self._refresh_info_labels()
        
        if self.worker:
            self.worker.update_bucket_edges(self.bucket_edges)
            self.status_label.setText("Applied bucket edges (live).")
        else:
            self.status_label.setText("Bucket edges updated (will apply on Start).")


    # ------------------------------------------------------------------
    # Colour mode: actuator colour + activation range
    # ------------------------------------------------------------------

    def on_actuator_colour_changed(self, actuator_id: int) -> None:
        name = self.colour_combos[actuator_id].currentText()
        self.actuator_colour_names[actuator_id] = name
        self._refresh_info_labels()
        if self.worker:
            self.worker.update_actuator_colour(actuator_id, name)

    def on_actuator_activation_changed(self, actuator_id: int) -> None:
        mn = int(self.min_duty_spins[actuator_id].value())
        mx = int(self.max_duty_spins[actuator_id].value())
        if mx < mn:
            mx = mn
            self.max_duty_spins[actuator_id].setValue(mx)

        self.actuator_min_duty[actuator_id] = mn
        self.actuator_max_duty[actuator_id] = mx
        self._refresh_info_labels()

        if self.worker:
            self.worker.update_actuator_activation(actuator_id, mn, mx)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def on_preset_changed(self, idx: int) -> None:
        name = self.preset_combo.itemText(idx)
        cfg = self.presets.get(name)
        if not cfg:
            return

        # Global freq
        gf = int(cfg.get("global_freq", self.current_global_freq_index))
        self.current_global_freq_index = gf
        self.global_freq_combo.blockSignals(True)
        self.global_freq_combo.setCurrentIndex(gf)
        self.global_freq_combo.blockSignals(False)

        # Per-actuator freqs
        freqs = cfg.get("freqs", [gf] * N_ACTUATORS)
        for i, combo in enumerate(self.freq_combos):
            val = int(freqs[i]) if i < len(freqs) else gf
            self.actuator_freq_indices[i] = val
            combo.blockSignals(True)
            combo.setCurrentIndex(val)
            combo.blockSignals(False)
            self.on_actuator_freq_changed(i, val)

        # Gains
        gains = cfg.get("gains", [1.0] * N_ACTUATORS)
        for i, spin in enumerate(self.gain_spins):
            g = float(gains[i]) if i < len(gains) else 1.0
            self.actuator_gains[i] = g
            spin.blockSignals(True)
            spin.setValue(g)
            spin.blockSignals(False)
            self.on_actuator_gain_changed(i, g)

        # Noise gate
        gate = float(cfg.get("noise_gate_db", self.current_noise_gate_db))
        self.current_noise_gate_db = gate
        self.noise_spin.blockSignals(True)
        self.noise_spin.setValue(gate)
        self.noise_spin.blockSignals(False)
        self.noise_label.setText(f"Noise gate (dB): {gate:.1f}")
        if self.worker:
            self.worker.update_noise_gate(gate)

        self.status_label.setText(f"Preset applied: {name}")

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------


    def show_help(self) -> None:
        text = """VibraForge / TimbreForge — GUI (Voice→Haptics) — guide rapide

Workflow conseillé (Phase 1)
  1) Hardware optionnel :
     - Si tu as le coeur VibraForge : laisse "No BLE" décoché (connexion BLE).
     - Si tu n'as PAS le hardware : coche "No BLE (visual-only)" et lance quand même.
  2) Calibration bruit ambiant : silence 1–2 s, puis "Calibrate".
  3) Parle / chante : observe les barres (Debug) et le feedback (haptics réels ou simulés).

Réglages importants
  • Spectral span (dB) : contraste/"compression" de l’énergie spectrale.
    - Plus petit = plus sensible (tout monte vite).
    - Plus grand = moins sensible, plus de contraste.

  • Dominance (grouped) : met en avant les K couleurs dominantes, sans supprimer les autres.
    - Top-K (1–3) : nombre de dominantes mises en avant.
    - Group size (1..3) : élargit la dominante à ses voisines (famille de couleurs).
    - Others gain (0..1) : contribution des couleurs non dominantes.
      (0.2–0.5 conseillé pour sentir une dominante tout en gardant le mélange.)

  • Noise subtract per-colour : soustraction du bruit par couleur après calibration.

Debug (barres)
  • Raw : avant calibration (brut).
  • Calibrated : après calibration (noise subtraction) + dominance (si activée).
    -> C’est la vue la plus proche de ce qui pilote les moteurs.

Tests rapides recommandés
  1) Silence (après calibration) : vibrations quasi nulles.
  2) Souffle : surtout Edge/Shimmer/Halo (selon ta voix / fuite d’air), pas tout à fond.
  3) Voix tenue ("aaaa") : stabilité (peu de jitter) + réponse graduelle.
  4) Forte vs faible : la duty doit augmenter clairement, sans tout saturer.
  5) A/B : Dominance OFF vs ON (Top-K=1, Others gain≈0.3) pour vérifier l’effet “dominante”.

Notes
  • Si "tout monte" quand tu souffles : augmente Spectral span (dB), baisse le gain micro, ou baisse Others gain.
  • Si rien ne bouge : baisse Spectral span (dB), vérifie micro/levels, refais calibration dans le silence.

"""
        #QMessageBox.information(self, "Aide / Guide", text)
        # Créer une fenêtre scrollable persistante (pas locale)
        from PyQt5.QtWidgets import QDialog
        
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Help - Vibraforge")
        help_dialog.setModal(True)
        help_layout = QVBoxLayout(help_dialog)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setFont(QFont("Courier", 9))
        text_label.setStyleSheet("padding: 10px;")
        
        scroll.setWidget(text_label)
        help_layout.addWidget(scroll)
        
        # Bouton de fermeture
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(help_dialog.close)
        help_layout.addWidget(close_btn)
        
        help_dialog.resize(700, 600)
        help_dialog.exec_()  # Utiliser exec_() pour bloquer jusqu'à fermeture

        # Accessibilité : imprime aussi dans la console (copiable)
        #print("\n" + text + "\n")


def main() -> None:
    app = QApplication(sys.argv)
    gui = VoiceHapticsGUI()
    gui.resize(1220, 900)
    gui.show()
    sys.exit(app.exec_())   


if __name__ == "__main__":
    main()