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
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread
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
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bucket_classification import make_bucket_config
from voice_colors import default_voice_color_channels
from voice_to_haptics_controller import VoiceToHapticsController


# -----------------------------------------------------------------------------
# GLOBAL GUI CONSTANTS
# -----------------------------------------------------------------------------

# Your current prototype: "9 actuators readily available, 6 in reality".
# The UI is written to be easily adjustable: change N_ACTUATORS here.
N_ACTUATORS = 6

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

    input_device_query: Optional[str]

class HapticsWorker(QObject):
    intensity_updated = pyqtSignal(object)  # array-like length N_ACTUATORS
    rms_updated = pyqtSignal(float, float, float, float)
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, cfg: RuntimeConfig):
        super().__init__()
        self.cfg = cfg
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.controller: Optional[VoiceToHapticsController] = None
        self.current_input_device_query = None


    # ---- Thread-safe commands from GUI (these schedule work inside asyncio loop) ----

    def stop(self, immediate_panic: bool = True) -> None:
        self._running = False
        if immediate_panic:
            self.panic_stop()

    def panic_stop(self) -> None:
        """Immediate stop, safe to call from GUI thread."""
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

        self.status.emit("Scanning/connecting Vibraforge BLE…")
        ok = await self.controller.connect_vibraforge()
        if not ok:
            self.status.emit("BLE device not found. Check power/name and retry.")
            raise RuntimeError("BLE device not found")

        self.status.emit("Connected. Opening microphone…")
        self.controller.open_microphone(device_name_contains=self.cfg.input_device_query)
        self.status.emit("Running. Speak/sing. ESC = PANIC STOP; Stop button = shutdown.")
        mic_was_ok = True
        try:
            while self._running:
                intensity = self.controller.process_audio_frame()

                if intensity is None:
                    if mic_was_ok:
                        mic_was_ok = False
                        # Stop haptics immediately if audio broke / stream closed
                        try:
                            await self.controller.panic_stop()
                        except Exception:
                            pass
                        self.status.emit("Microphone error/closed -> haptics stopped.")
                    await asyncio.sleep(0.05)
                    continue
                else:
                    if not mic_was_ok:
                        mic_was_ok = True
                        self.status.emit("Microphone OK -> resuming.")

                # Update GUI
                self.intensity_updated.emit(intensity.copy())

                # Drive haptics
                await self.controller.send_haptic_command(intensity)

                # RMS stats
                mean5, peak5, mean10, peak10 = self.controller.get_rms_stats()
                self.rms_updated.emit(mean5, peak5, mean10, peak10)

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
        self.current_global_freq_index: int = 5
        self.current_input_device_query: Optional[str] = None


        # Current per-actuator configuration
        self.actuator_freq_indices = [self.current_global_freq_index] * N_ACTUATORS
        self.actuator_gains = [1.0] * N_ACTUATORS
        self.actuator_min_duty = [0] * N_ACTUATORS
        self.actuator_max_duty = [15] * N_ACTUATORS

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

        # Overall scroll (so the UI can go right-first + still be navigable vertically)
        outer = QVBoxLayout(w)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)

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

        # ========== CONTROL STRIP (right-first layout) ==========
        control_strip = QGroupBox("Session / Global controls")
        cs = QGridLayout(control_strip)
        cs.setHorizontalSpacing(10)
        cs.setVerticalSpacing(6)
        cs.setContentsMargins(10, 10, 10, 10)

        # Mode selector
        cs.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Normal (spectral buckets)", userData="normal")
        self.mode_combo.addItem("Colour (timbre channels)", userData="colour")
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        cs.addWidget(self.mode_combo, 0, 1)

        cs.addWidget(QLabel("Audio input:"), 2, 0)
        self.input_combo = QComboBox()
        self.input_combo.addItem("Default (Windows)", userData=None)
        self.input_combo.addItem("Raw headset mic (Realtek)", userData="Headset Microphone")
        self.input_combo.addItem("OBS RNNoise (VB-CABLE)", userData="CABLE Output")
        self.input_combo.currentIndexChanged.connect(self.on_input_device_changed)
        cs.addWidget(self.input_combo, 2, 1, 1, 3)
        self.current_input_device_query = self.input_combo.currentData()


        # Start/Stop
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        cs.addWidget(self.start_btn, 0, 2)
        cs.addWidget(self.stop_btn, 0, 3)

        # Global frequency
        cs.addWidget(QLabel("Global vibration freq index (0–7):"), 1, 0)
        self.global_freq_combo = QComboBox()
        for i in range(8):
            self.global_freq_combo.addItem(str(i))
        self.global_freq_combo.setCurrentIndex(self.current_global_freq_index)
        self.global_freq_combo.currentIndexChanged.connect(self.on_global_freq_changed)
        cs.addWidget(self.global_freq_combo, 1, 1)

        # Noise gate
        self.noise_label = QLabel(f"Noise gate (dB): {self.current_noise_gate_db:.1f}")
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(-80.0, -20.0)
        self.noise_spin.setSingleStep(1.0)
        self.noise_spin.setValue(self.current_noise_gate_db)
        self.noise_spin.valueChanged.connect(self.on_noise_gate_changed)
        cs.addWidget(self.noise_label, 1, 2)
        cs.addWidget(self.noise_spin, 1, 3)

        root.addWidget(control_strip)

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

        for i in range(N_ACTUATORS):
            r = i + 1

            act_lbl = QLabel(f"{i}")
            act_lbl.setMinimumWidth(act_label_w)
            grid.addWidget(act_lbl, r, 0)

            # Color intensity box
            color_box = QWidget()
            color_box.setMinimumHeight(24)
            color_box.setMinimumWidth(color_w)
            color_box.setStyleSheet(f"background-color: {INTENSITY_PALETTE[0]}; border-radius: 4px; border: 1px solid #333333;")
            self.intensity_color_boxes.append(color_box)
            grid.addWidget(color_box, r, 1)

            # Intensity bar + text
            main = QLabel(f"[{bar(0)}] 0/15")
            main.setFont(font_bar)
            self.intensity_labels.append(main)
            grid.addWidget(main, r, 2)

            # Per-actuator freq
            combo = QComboBox()
            for k in range(8):
                combo.addItem(str(k))
            combo.setCurrentIndex(self.actuator_freq_indices[i])
            combo.currentIndexChanged.connect(lambda idx, a=i: self.on_actuator_freq_changed(a, idx))
            self.freq_combos.append(combo)
            grid.addWidget(combo, r, 3)

            # Per-actuator gain
            gain_spin = QDoubleSpinBox()
            gain_spin.setRange(0.0, 3.0)
            gain_spin.setSingleStep(0.1)
            gain_spin.setValue(self.actuator_gains[i])
            gain_spin.valueChanged.connect(lambda val, a=i: self.on_actuator_gain_changed(a, val))
            self.gain_spins.append(gain_spin)
            grid.addWidget(gain_spin, r, 4)

            # Activation range (always visible; useful in both modes)
            mn = QSpinBox()
            mn.setRange(0, 15)
            mn.setValue(self.actuator_min_duty[i])
            mn.valueChanged.connect(lambda _v, a=i: self.on_actuator_activation_changed(a))
            self.min_duty_spins.append(mn)
            grid.addWidget(mn, r, 5)

            mx = QSpinBox()
            mx.setRange(0, 15)
            mx.setValue(self.actuator_max_duty[i])
            mx.valueChanged.connect(lambda _v, a=i: self.on_actuator_activation_changed(a))
            self.max_duty_spins.append(mx)
            grid.addWidget(mx, r, 6)

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
            grid.addWidget(lo, r, 7)
            grid.addWidget(hi, r, 8)

            # Colour-mode combo
            ccombo = QComboBox()
            for name in self.colour_names:
                ccombo.addItem(name)
            ccombo.setCurrentText(self.actuator_colour_names[i])
            ccombo.currentIndexChanged.connect(lambda _idx, a=i: self.on_actuator_colour_changed(a))
            self.colour_combos.append(ccombo)
            grid.addWidget(ccombo, r, 9)

            # Info label (kept on the same row to avoid “drifting distance”)
            info = QLabel("–")
            info.setFont(font_info)
            info.setStyleSheet("color: #888888;")
            info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.info_labels.append(info)
            grid.addWidget(info, r, 10)

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
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        runtime_cfg = RuntimeConfig(
            mode=self.current_mode,
            noise_gate_db=self.current_noise_gate_db,
            global_freq_index=self.current_global_freq_index,
            bucket_edges=list(self.bucket_edges),
            actuator_colour_names=list(self.actuator_colour_names),
            actuator_freq_indices=list(self.actuator_freq_indices),
            actuator_gains=list(self.actuator_gains),
            actuator_min_duty=list(self.actuator_min_duty),
            actuator_max_duty=list(self.actuator_max_duty),
            input_device_query=self.current_input_device_query,
        )

        self.thread = QThread()
        self.worker = HapticsWorker(runtime_cfg)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.intensity_updated.connect(self.on_intensity)
        self.worker.status.connect(self.status_label.setText)
        self.worker.rms_updated.connect(self.on_rms_updated)
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
        # Header indices are defined in the UI build: Bucket lo=7, Bucket hi=8, Colour=9.
        if hasattr(self, "_hdr_widgets") and len(self._hdr_widgets) >= 10:
            self._hdr_widgets[7].setVisible(is_normal)
            self._hdr_widgets[8].setVisible(is_normal)
            self._hdr_widgets[9].setVisible(is_colour)

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

    def on_actuator_gain_changed(self, actuator_id: int, value: float) -> None:
        self.actuator_gains[actuator_id] = float(value)
        if self.worker:
            self.worker.update_actuator_gain(actuator_id, float(value))

    def on_noise_gate_changed(self, value: float) -> None:
        self.current_noise_gate_db = float(value)
        self.noise_label.setText(f"Noise gate (dB): {self.current_noise_gate_db:.1f}")
        if self.worker:
            self.worker.update_noise_gate(self.current_noise_gate_db)

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
        text = (
            "Vibraforge Voice→Haptics GUI – quick tour\n\n"
            "Start/Stop\n"
            "  • Start connects BLE, opens microphone, and begins real-time mapping.\n"
            "  • Stop performs a clean shutdown (releases BLE + mic so you can restart).\n"
            "  • ESC triggers PANIC STOP (immediate motor stop).\n\n"
            "Modes\n"
            "  • Normal mode: you set frequency bucket boundaries in Hz.\n"
            "    Each bucket drives the same-index actuator.\n"
            "  • Colour mode: you assign named vocal ‘colours’ to actuators.\n"
            "    Multiple actuators can share one colour.\n\n"
            "Per-actuator controls\n"
            "  • freq index (0–7): chooses the vibrotactile carrier preset on the hardware.\n"
            "  • gain (0–3): multiplies intensity after analysis (useful to balance tactors).\n"
            "  • active duty: vibration is suppressed below min; capped at max.\n\n"
            "Noise gate\n"
            "  • If the signal RMS is below the gate, all motors are off.\n"
            "  • Raise gate if you get buzzing in silence.\n\n"
            "Bucket templates\n"
            "  • Voice-landmark is recommended for voice learning with few actuators.\n"
            "  • ERB/Mel/Log/Linear are useful for benchmarking or exploratory mappings.\n"
        )

        QMessageBox.information(self, "Help", text)


def main() -> None:
    app = QApplication(sys.argv)
    gui = VoiceHapticsGUI()
    gui.resize(1220, 900)
    gui.show()
    sys.exit(app.exec_())   


if __name__ == "__main__":
    main()
