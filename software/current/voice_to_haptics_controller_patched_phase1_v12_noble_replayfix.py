# voice_to_haptics_controller.py
"""================================================================================
Vibraforge – Voice → Haptics Controller (Realtime)
================================================================================

Purpose
-------
This file is the real-time "engine" that:
1) Captures microphone audio (PyAudio)
2) Computes spectral features (bucket_classification.py)
3) Converts features -> actuator intensity (0..15)
4) Sends Vibraforge commands over BLE (Bleak) to an ESP32 controller

Two analysis modes are supported
--------------------------------
The project uses TWO complementary representations of voice:

A) NORMAL MODE (spectral buckets)
   - You define M frequency buckets in Hz (M usually equals the number of
     available actuators for a simple PoC).
   - We compute bucket energies b[m] (per bucket) and convert them to intensities.
   - Intuition: "this actuator = this frequency band".

B) COLOUR MODE (timbre 'colours')
   - We compute a stable **canonical** representation of the spectrum
     (K=32 ERB bands by default).
   - We define a set of named colour channels (Warmth / Presence / Shimmer / ...)
     as weighted combinations of the canonical bands (voice_colors.py).
   - Each actuator is assigned to ONE colour channel (multiple actuators can share
     the same colour).
   - Intuition: "this actuator = this vocal quality descriptor".

Why a two-layer design?
-----------------------
Both modes share the same canonical front-end. Only the low-dimensional mapping
changes. This keeps experiments comparable and makes it safe to change the number
of available actuators:

    frame -> canonical energies e[K]  (stable)
         -> buckets b[M]             (normal)
         -> colours c[C]             (colour)
         -> actuator duties a[N]

Protocol / Firmware notes
-------------------------
- Vibraforge command is 3 bytes per actuator command.
- Packets are 60 bytes (20 commands) to match your existing working firmware.
- We avoid "stacking" by optionally sending STOP(old) before START(new) whenever
  duty or frequency changes while an actuator is running.

Troubleshooting note: "Stop then Start doesn't work unless I restart the program"
--------------------------------------------------------------------------------
If BLE or audio resources are not released cleanly, restarting can fail.
This file provides a dedicated async shutdown() method that:
- sends a stop to all mapped actuators
- closes the audio stream
- disconnects the BLE client
- stops the Windows BLE MTA thread (if used)

The GUI should call shutdown() when the worker stops.
================================================================================
"""

from __future__ import annotations
from typing import Optional

import sys
import threading
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyaudio
from bleak import BleakClient, BleakScanner

from bucket_classification import (
    make_bucket_config,
    compute_bucket_energies,
    compute_canonical_energies,
    update_bucket_edges,
)
from voice_colors import default_voice_color_channels, build_color_matrix

# -----------------------------------------------------------------------------
# Optional dependency: soundfile (for easy float32 WAV I/O). We provide fallbacks.
# -----------------------------------------------------------------------------
try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None

import wave
import os

from datetime import datetime

BLE_LOG_PATH = os.path.join(os.path.dirname(__file__), "vibraforge_ble_scan.log")

def _ble_log(msg: str) -> None:
    """Append a timestamped line to vibraforge_ble_scan.log (best effort)."""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with open(BLE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Audio source abstraction (Live / Record / Replay)
# -----------------------------------------------------------------------------

class BaseAudioSource:
    """Abstraction so the controller can run with:
    - live microphone
    - microphone + recording to WAV
    - WAV playback (optionally to speaker) + haptics

    read(n) returns float32 mono samples of length n, or None when finished.
    """

    def read(self, n_frames: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MicAudioSource(BaseAudioSource):
    def __init__(self, stream: 'pyaudio.Stream'):
        self.stream = stream

    def read(self, n_frames: int) -> Optional[np.ndarray]:
        try:
            data = self.stream.read(n_frames, exception_on_overflow=False)
        except OSError:
            return None
        return np.frombuffer(data, dtype=np.float32)


class RecordingAudioSource(BaseAudioSource):
    def __init__(self, mic: MicAudioSource, *, record_path: str, sr: int):
        self.mic = mic
        self.record_path = record_path
        self.sr = int(sr)
        self._frames: list[np.ndarray] = []
        self._sf = None
        if sf is not None:
            try:
                self._sf = sf.SoundFile(record_path, mode='w', samplerate=self.sr, channels=1, subtype='FLOAT')
            except Exception:
                self._sf = None

    def read(self, n_frames: int) -> Optional[np.ndarray]:
        x = self.mic.read(n_frames)
        if x is None:
            return None
        try:
            if self._sf is not None:
                self._sf.write(x.reshape(-1, 1))
            else:
                self._frames.append(x.copy())
        except Exception:
            pass
        return x

    def close(self) -> None:
        try:
            if self._sf is not None:
                self._sf.close()
                self._sf = None
                return
        except Exception:
            pass
        if self._frames:
            data = np.concatenate(self._frames).astype(np.float32)
            pcm = np.clip(data, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            with wave.open(self.record_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sr)
                wf.writeframes(pcm16.tobytes())
        self._frames.clear()


def _linear_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if len(x) < 2:
        return np.zeros(int(round(len(x) * sr_out / sr_in)), dtype=np.float32)
    ratio = sr_out / sr_in
    n_out = int(round(len(x) * ratio))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


class PlaybackAudioSource(BaseAudioSource):
    def __init__(
        self,
        *,
        pya: 'pyaudio.PyAudio',
        playback_path: str,
        target_sr: int,
        chunk_size: int,
        to_speaker: bool = False,
    ):
        self.pya = pya
        self.playback_path = playback_path
        self.target_sr = int(target_sr)
        self.chunk_size = int(chunk_size)
        self.to_speaker = bool(to_speaker)

        self._pos = 0
        self._out_stream = None

        data, sr = self._load_audio(playback_path)
        if sr != self.target_sr:
            data = _linear_resample(data, sr, self.target_sr)
        self.data = data.astype(np.float32, copy=False)

        if self.to_speaker:
            try:
                self._out_stream = self.pya.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.target_sr,
                    output=True,
                    frames_per_buffer=self.chunk_size,
                )
            except Exception:
                self._out_stream = None

    def _load_audio(self, path: str) -> tuple[np.ndarray, int]:
        if sf is not None:
            try:
                x, sr = sf.read(path, dtype='float32', always_2d=False)
                if hasattr(x, 'ndim') and x.ndim > 1:
                    x = x[:, 0]
                return np.asarray(x, dtype=np.float32), int(sr)
            except Exception:
                pass
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            frames = wf.readframes(n)
            sw = wf.getsampwidth()
            if sw == 2:
                x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                x = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                x = (x - 128.0) / 128.0
            if wf.getnchannels() > 1:
                x = x.reshape(-1, wf.getnchannels())[:, 0]
            return x.astype(np.float32), int(sr)

    def read(self, n_frames: int) -> Optional[np.ndarray]:
        if self._pos >= len(self.data):
            return None
        end = min(self._pos + n_frames, len(self.data))
        chunk = self.data[self._pos:end]
        self._pos = end
        if len(chunk) < n_frames:
            chunk = np.pad(chunk, (0, n_frames - len(chunk)))
        if self._out_stream is not None:
            try:
                self._out_stream.write(chunk.astype(np.float32).tobytes())
            except Exception:
                pass
        return chunk.astype(np.float32, copy=False)

    def close(self) -> None:
        if self._out_stream is not None:
            try:
                self._out_stream.stop_stream()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None


# -----------------------------------------------------------------------------
# BLE constants – change if your Vibraforge firmware changes
# -----------------------------------------------------------------------------

CHARACTERISTIC_UUID = "f22535de-5375-44bd-8ca9-d0ea9ff9e410"

# BLE name handling (robustness)
# On Windows (WinRT), advertised device names can be missing or inconsistent.
# We therefore try a few known names first, then fall back to:
#   scan -> connect -> verify that our characteristic UUID exists.
CONTROL_UNIT_NAME = "QT Py ESP32-S3"
CONTROL_UNIT_NAME_CANDIDATES = [
    "QT Py ESP32-S3",
    "Vibraforge",
    "VibraForge",
    "VibraForge-Core",
    "Vibraforge-Core",
    "TimbreForge",
]
CONTROL_UNIT_NAME_TOKENS = ["vibra", "forge", "esp32", "qt", "timbre"]

MAX_COMMANDS_PER_PACKET = 20
PACKET_SIZE_BYTES = 60

# ----------------------------------------------------------------------------
# Addressing helper (Phase 1): map (channel, address) -> physical actuator id
# ----------------------------------------------------------------------------

# NOTE:
# Your firmware command packs a single integer 'addr' into (group, serial_addr).
# We expose a friendly (channel, address) pair in the GUI and translate it here.
# By default we assume 16 addresses per channel/group because create_command()
# currently uses: group = addr // 16, serial_addr = addr % 16.
ADDRS_PER_CHANNEL = 16
MAX_CHANNELS = 8

@dataclass(frozen=True)
class ChannelAddress:
    channel: int
    address: int

    def to_physical_id(self, addrs_per_channel: int = ADDRS_PER_CHANNEL) -> int:
        return int(self.channel) * int(addrs_per_channel) + int(self.address)



# -----------------------------------------------------------------------------
# Windows-specific: ensure Bleak runs on an MTA thread (WinRT requirement)
# -----------------------------------------------------------------------------

class _BleakWinMtaThread:
    """Run Bleak WinRT operations on a dedicated MTA thread.

    Why:
    - On Windows, Bleak's WinRT backend depends on a message pump / callback
      mechanism that behaves correctly only when executed in an MTA context.
    - Console apps can end up STA by default, leading to connect/scan issues.

    How it works:
    - Create an asyncio event loop inside a daemon thread
    - Schedule all Bleak coroutines onto that loop

    You normally do not need to change this.
    """

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.loop is not None:
            return
        self.loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(target=_runner, daemon=True)
        self.thread.start()

    async def run(self, coro):
        """Schedule a coroutine onto the BLE thread loop and await its result."""
        if self.loop is None:
            raise RuntimeError("BLE thread not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return await asyncio.wrap_future(fut)

    def stop(self) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop = None
        self.thread = None


# -----------------------------------------------------------------------------
# Mapping: bucket/channel index -> actuator addresses
# -----------------------------------------------------------------------------

@dataclass
class ActuatorMapping:
    """Route a logical channel (bucket index) to one or more actuator addresses."""

    bucket_id: int
    actuator_ids: List[int]
    body_location: str = ""


def default_actuator_mapping_identity(n: int) -> List[ActuatorMapping]:
    """Default: channel i -> actuator i (one-to-one)."""
    return [ActuatorMapping(bucket_id=i, actuator_ids=[i], body_location=f"A{i}") for i in range(n)]


# -----------------------------------------------------------------------------
# Intensity mapping: energies -> duty levels 0..max_duty
# -----------------------------------------------------------------------------

def energies_to_intensity(
    channel_energies: np.ndarray,
    rms_db: float,
    noise_gate_db: float,
    *,
    max_duty: int = 12,
    loudness_span_db: float = 20.0,
    spectral_span_db: float = 25.0,
    gamma: float = 1.0,
    bias_db: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert energies -> duty levels (0..max_duty).

    This converts an energy vector (one value per bucket or per colour channel)
    into a small integer duty suitable for Vibraforge.

    The conversion uses two ideas:

    1) Global loudness gate (RMS dB)
       - If the signal is quiet, we output 0 (no vibration).
       - As RMS increases above the gate, all duties scale up.

    2) Spectral contrast
       - Within a frame, channels near the strongest channel are emphasized.
       - Channels far below the maximum are suppressed.

    Parameters
    ----------
    channel_energies:
        Non-negative energy values per channel.

    rms_db:
        RMS dB of the full frame.

    noise_gate_db:
        Silence threshold. If rms_db < noise_gate_db -> all zeros.

    max_duty:
        Maximum duty to output (<=15).

    loudness_span_db:
        dB range above gate that maps from 0 -> full loudness.

    spectral_span_db:
        Dynamic range within the spectrum. Anything more than this below the
        per-frame peak is treated as off.

    bias_db:
        Optional per-channel dB offsets. Useful if you want to slightly
        favor lows vs highs (normal mode). If None -> no bias.

    Returns
    -------
    duties:
        uint8 array of the same length as channel_energies.
    """

    eps = 1e-10
    e = np.maximum(channel_energies, 0.0) + eps

    # 0) Noise gate
    if rms_db < noise_gate_db:
        return np.zeros_like(e, dtype=np.uint8)

    # 1) Loudness factor g in [0, 1]
    g = (rms_db - noise_gate_db) / float(loudness_span_db)
    g = float(np.clip(g, 0.0, 1.0))

    # 2) Spectral contrast s in [0, 1]
    ch_db = 10.0 * np.log10(e)  # energies are power-like
    if bias_db is not None:
        bias_db = np.asarray(bias_db, dtype=np.float32)
        if len(bias_db) >= len(ch_db):
            ch_db = ch_db + bias_db[: len(ch_db)]
        else:
            ch_db = ch_db + np.pad(bias_db, (0, len(ch_db) - len(bias_db)), constant_values=0.0)

    max_db = float(np.max(ch_db))
    if not np.isfinite(max_db):
        return np.zeros_like(e, dtype=np.uint8)

    floor = max_db - float(spectral_span_db)
    s = (ch_db - floor) / (max_db - floor + 1e-6)
    s[ch_db <= floor] = 0.0
    s = np.clip(s, 0.0, 1.0)

    x = np.clip((g * s), 0.0, 1.0)
    if gamma and float(gamma) != 1.0:
        x = x ** float(gamma)

    duties = np.round(x * float(max_duty)).astype(np.uint8)
    return duties


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------

class VoiceToHapticsController:
    """Realtime voice -> haptics controller.

    The controller is deliberately "headless": it does not depend on Qt.
    It is meant to be driven by a worker thread (e.g., in voice_haptics_gui.py).
    """

    def __init__(
        self,
        *,
        # Audio
        sr: int = 48000,
        n_fft: int = 2048,
        chunk_size: int = 1024,
        # Spectral analysis
        fmin: float = 50.0,
        fmax: float = 12000.0,
        canonical_K: int = 32,
        canonical_scale: str = "erb",
        # Actuator-side "normal" buckets
        n_actuators: int = 6,
        bucket_method: str = "voice_landmark",
        inner_scale: str = "erb",
        # Vibraforge
        freq_index: int = 5,
        reset_on_change: bool = True,
        idle_stop_timeout_s: float = 0.30,
        # Loudness / sensitivity
        noise_gate_db: float = -60.0,
    ):
        # ---------------- BLE / platform ----------------
        self._ble_thread = _BleakWinMtaThread() if sys.platform == "win32" else None

        # ---------------- Analysis configuration ----------------
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.chunk_size = int(chunk_size)

        self.fmin = float(fmin)
        self.fmax = float(fmax)

        self.n_actuators = int(n_actuators)
        self.bucket_method = str(bucket_method)
        self.inner_scale = str(inner_scale)

        # Current mode: "normal" or "color"
        self.mode: str = "normal"

                # Vibraforge
        self.freq_index = int(freq_index) & 0x07
        self.reset_on_change = bool(reset_on_change)
        self.idle_stop_timeout_s = float(idle_stop_timeout_s)

        # Loudness gating
        self.noise_gate_db = float(noise_gate_db)

        # Canonical + bucket config
        # NOTE: M equals the number of actuators in the baseline PoC.
        self.cfg = make_bucket_config(
            sr=self.sr,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax,
            K=int(canonical_K),
            canonical_scale=canonical_scale,  # recommended "erb"
            M=self.n_actuators,
            bucket_method=self.bucket_method,
            inner_scale=self.inner_scale,
        )
        
    
        # --------- Actuator routing (logical bucket -> physical actuator id) ---------
        # By default we keep the old behaviour: logical actuator i -> physical id i.
        # Physical ids are what create_command() expects.
        self.mapping = default_actuator_mapping_identity(self.n_actuators)
        self._rebuild_routing()

        # --------- Per-actuator controls (live-tunable from GUI) ---------
        # Gain scales the computed duty (after mapping), then we clip to 0..15.
        self.actuator_gains = np.ones(self.n_actuators, dtype=np.float32)

        # Per-actuator frequency index (0..7). Default = global freq_index.
        self.freq_per_bucket = [self.freq_index] * self.n_actuators

        # Per-actuator activation range in duty units.
        # Example: min=2 means ignore tiny duties (reduces buzzing);
        # max can be used to cap strong vibrations.
        self.min_duty_per_act = np.zeros(self.n_actuators, dtype=np.uint8)
        self.max_duty_per_act = (np.ones(self.n_actuators, dtype=np.uint8) * 15)

        # --------- Colour channels (used only in colour mode) ---------
        self.color_channels = default_voice_color_channels(self.fmin, self.fmax)
        self._color_name_to_idx = {c.name: i for i, c in enumerate(self.color_channels)}

        canonical_centers = self.cfg.canonical_edges_hz[1:-1]
        self.color_matrix = build_color_matrix(
            canonical_centers_hz=canonical_centers,
            channels=self.color_channels,
            normalize="mean",
        )

        # Each actuator selects one colour channel by index.
        # Default: first N colours (or wrap if fewer colours than actuators).
        self.actuator_color_idx = np.array(
            [i % len(self.color_channels) for i in range(self.n_actuators)],
            dtype=np.int32,
        )

        # --------- State for "non-additive reset" (avoid stacking) ---------
        self._last_state: Dict[int, Tuple[int, int, int]] = {}  # addr -> (mode, freq, duty)
        self._last_ok_time = time.monotonic()

        # --------- Audio I/O ---------
        self._pyaudio: Optional[pyaudio.PyAudio] = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

        # Audio source abstraction (selected by GUI/worker)
        self.audio_source: Optional[BaseAudioSource] = None
        self.audio_mode: str = "live"

        # Per-colour noise floor subtraction (helps avoid persistent 'warmth' etc.)
        self.noise_floor_colour: Optional[np.ndarray] = None
        self.colour_noise_subtract_enabled: bool = True
        self.colour_noise_subtract_alpha: float = 1.0


        # Live robustness (casual rooms): suppress unvoiced breath / broadband noise.
        self.voicedness_gate_enabled: bool = True
        self.voiced_flatness_floor: float = 0.28   # <= -> strongly voiced
        self.voiced_flatness_thresh: float = 0.60  # >= -> mostly noise-like
        self.voicedness_min: float = 0.25          # do not fully mute unvoiced content
        self.loudness_gamma: float = 2.2           # >1 reduces sensitivity near the gate
        self.last_spectral_flatness: float = float("nan")
        self.last_voicedness: float = 1.0

        # Spectral normalization / shaping
        # - spectral_span_db: larger -> more channels active (more sensitive); smaller -> sparser activations.
        self.spectral_span_db: float = 25.0

        # Colour-mode dominance (grouped, non-destructive)
        # When enabled, we emphasise the top-K colour channels (optionally expanding to neighbours),
        # while still keeping a (reduced) contribution from all colours.
        self.dominance_enabled: bool = False
        self.dominance_top_k: int = 2          # how many dominant colours to emphasise
        self.dominance_group_size: int = 1     # 1..3 (dominant alone / +1 neighbour / +/-1 neighbours)
        self.dominance_non_dominant_gain: float = 0.35  # 0..1 gain applied to non-dominant colours

        # BLE debug
        self.last_ble_scan: List[Tuple[Optional[str], str]] = []
        self.last_ble_error: Optional[str] = None
        self.ble_connected_name: Optional[str] = None
        self.ble_connected_address: Optional[str] = None
        # Latest debug features (for GUI bargraphs)
        self.last_rms_db: float = float("nan")
        self.last_bucket_energies: Optional[np.ndarray] = None
        self.last_canonical_energies: Optional[np.ndarray] = None
        self.last_colour_energies_raw: Optional[np.ndarray] = None
        self.last_colour_energies_clean: Optional[np.ndarray] = None
        self.last_colour_energies_drive: Optional[np.ndarray] = None
        self.audio_buffer = np.zeros(self.n_fft, dtype=np.float32)

        # --------- BLE ---------
        self.client: Optional[BleakClient] = None
        self.is_connected: bool = False

        # --------- Loudness history for GUI ---------
        self.rms_history: List[Tuple[float, float]] = []
        self.max_history_s: float = 10.0

    # -----------------------------------------------------------------
    # Public configuration API (called by GUI)
    # -----------------------------------------------------------------


    # -----------------------------------------------------------------
    # Actuator address mapping (Phase 1)
    # -----------------------------------------------------------------
    def _rebuild_routing(self) -> None:
        """Recompute derived routing tables after any mapping change."""
        self.mapped_actuators = sorted({a for m in self.mapping for a in m.actuator_ids})
        # physical_id -> logical bucket_id (for freq selection)
        self.actuator_to_bucket = {}
        for m in self.mapping:
            for phys in m.actuator_ids:
                self.actuator_to_bucket[int(phys)] = int(m.bucket_id)

    @staticmethod
    def channel_address_to_physical_id(channel: int, address: int, *, addrs_per_channel: int = ADDRS_PER_CHANNEL) -> int:
        return int(channel) * int(addrs_per_channel) + int(address)

    def set_actuator_channel_address(self, actuator_id: int, channel: int, address: int) -> None:
        """Assign a logical actuator (bucket) to a (channel, address)."""
        a = int(actuator_id)
        if a < 0 or a >= len(self.mapping):
            return
        ch = int(channel)
        ad = int(address)
        ch = max(0, min(ch, MAX_CHANNELS - 1))
        ad = max(0, min(ad, ADDRS_PER_CHANNEL - 1))
        phys = self.channel_address_to_physical_id(ch, ad)
        # One-to-one in Phase 1: each logical actuator drives exactly one physical address.
        self.mapping[a].actuator_ids = [phys]
        self._rebuild_routing()

    def set_all_actuator_channel_addresses(self, channels: List[int], addresses: List[int]) -> None:
        """Bulk update mapping for all logical actuators."""
        n = min(len(channels), len(addresses), len(self.mapping))
        for a in range(n):
            ch = int(channels[a])
            ad = int(addresses[a])
            ch = max(0, min(ch, MAX_CHANNELS - 1))
            ad = max(0, min(ad, ADDRS_PER_CHANNEL - 1))
            phys = self.channel_address_to_physical_id(ch, ad)
            self.mapping[a].actuator_ids = [phys]
        self._rebuild_routing()

    def set_mode(self, mode: str) -> None:
        """Set analysis mode: 'normal' or 'color'."""
        mode = mode.strip().lower()
        if mode not in {"normal", "color"}:
            raise ValueError("mode must be 'normal' or 'color'")
        self.mode = mode

    def set_noise_gate(self, noise_gate_db: float) -> None:
        """Set silence threshold in dB."""
        self.noise_gate_db = float(noise_gate_db)
    def set_colour_noise_floor(self, noise_floor_colour: Optional[np.ndarray]) -> None:
        """Set per-colour noise floor (C,) used for subtraction in colour mode."""
        if noise_floor_colour is None:
            self.noise_floor_colour = None
            return
        v = np.asarray(noise_floor_colour, dtype=np.float32).reshape(-1)
        self.noise_floor_colour = v

    def set_colour_noise_subtract(self, enabled: bool, alpha: float = 1.0) -> None:
        """Enable/disable per-colour noise subtraction and set aggressiveness."""
        self.colour_noise_subtract_enabled = bool(enabled)
        self.colour_noise_subtract_alpha = float(alpha)

    def set_spectral_span_db(self, spectral_span_db: float) -> None:
        """Set the spectral dynamic range (dB) used for per-frame normalization.

        Larger values make the system more sensitive (more bands/colours become nonzero).
        Smaller values make activations sparser.
        """
        self.spectral_span_db = float(spectral_span_db)

    def set_colour_dominance(
        self,
        enabled: bool,
        top_k: int = 2,
        group_size: int = 1,
        non_dominant_gain: float = 0.35,
    ) -> None:
        """Enable/disable colour dominance and set its parameters.

        This dominance is intentionally **non-destructive**: instead of hard-muting non-dominant colours,
        we attenuate them by `non_dominant_gain` so the overall colour mixture is preserved.

        Parameters
        ----------
        enabled:
            Whether dominance is active.
        top_k:
            Number of dominant colours to emphasise (clamped to >=1).
        group_size:
            Dominance group size (1..3). 1=only the dominant colour, 2=dominant + one neighbour,
            3=dominant +/- one neighbour.
        non_dominant_gain:
            Gain applied to non-dominant colours (0..1). Lower = more 'one colour stands out'.
        """
        self.dominance_enabled = bool(enabled)
        self.dominance_top_k = int(max(1, top_k))
        self.dominance_group_size = int(max(1, min(3, group_size)))
        try:
            g = float(non_dominant_gain)
        except Exception:
            g = 0.35
        self.dominance_non_dominant_gain = float(max(0.0, min(1.0, g)))

    def _apply_colour_dominance(self, c: np.ndarray) -> np.ndarray:
        """Apply dominance weighting to colour energies (returns a *new* array)."""
        c_drive = np.asarray(c, dtype=np.float32).reshape(-1)
        if not bool(self.dominance_enabled):
            return c_drive

        n = int(c_drive.shape[0])
        if n <= 0:
            return c_drive

        try:
            k = int(self.dominance_top_k)
        except Exception:
            k = 2
        k = max(1, min(k, n))

        try:
            group_size = int(self.dominance_group_size)
        except Exception:
            group_size = 1
        group_size = max(1, min(3, group_size))

        try:
            non_dom_gain = float(self.dominance_non_dominant_gain)
        except Exception:
            non_dom_gain = 0.35
        non_dom_gain = float(max(0.0, min(1.0, non_dom_gain)))

        # Pick top-K dominant colours
        idx_sorted = np.argsort(c_drive)[::-1]
        chosen = idx_sorted[:k]

        # Expand to neighbours depending on group size
        boost = np.zeros(n, dtype=bool)
        for ii in chosen:
            i = int(ii)
            if not (0 <= i < n):
                continue
            boost[i] = True
            if group_size >= 2:
                if group_size == 2:
                    left = i - 1 if (i - 1) >= 0 else None
                    right = i + 1 if (i + 1) < n else None
                    if left is None and right is not None:
                        boost[right] = True
                    elif right is None and left is not None:
                        boost[left] = True
                    elif left is not None and right is not None:
                        # pick the stronger neighbour
                        if c_drive[left] >= c_drive[right]:
                            boost[left] = True
                        else:
                            boost[right] = True
                else:  # group_size == 3
                    if i - 1 >= 0:
                        boost[i - 1] = True
                    if i + 1 < n:
                        boost[i + 1] = True

        # Non-destructive weighting: attenuate non-dominant colours
        weights = np.ones(n, dtype=np.float32) * np.float32(non_dom_gain)
        weights[boost] = np.float32(1.0)
        return c_drive * weights

    def get_debug_meta(self) -> dict:
        """Metadata bundle for GUI debug bars."""
        try:
            centers = getattr(self.cfg, "canonical_centers_hz", None)
            centers_list = centers.tolist() if centers is not None else None
        except Exception:
            centers_list = None
        return {
            "canonical_centers_hz": centers_list,
            "colour_names": [c.name for c in self.color_channels],
        }

    def compute_debug_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (canonical, colour_raw, colour_clean) from the current audio_buffer.

        This is meant to be called at a throttled rate by the GUI/worker to draw bargraphs,
        without doubling FFT work every single frame.
        """
        e = compute_canonical_energies(self.audio_buffer, self.cfg).astype(np.float32, copy=False)
        c_raw = (self.color_matrix @ e).astype(np.float32, copy=False)

        c = c_raw
        if self.colour_noise_subtract_enabled and (self.noise_floor_colour is not None):
            floor = np.asarray(self.noise_floor_colour, dtype=np.float32).reshape(-1)
            if floor.shape[0] == c_raw.shape[0]:
                c = np.maximum(0.0, c_raw - (self.colour_noise_subtract_alpha * floor))

        # store for convenience
        self.last_canonical_energies = e
        self.last_colour_energies_raw = c_raw
        self.last_colour_energies_clean = c.astype(np.float32, copy=False)
        c_drive = self._apply_colour_dominance(self.last_colour_energies_clean)
        self.last_colour_energies_drive = c_drive.astype(np.float32, copy=False)
        return e, c_raw, self.last_colour_energies_drive

    def calibrate_room_noise(
    self,
    *,
    duration_s: float = 2.0,
    percentile: float = 75.0,
    gate_margin_db: float = 8.0,
    colour_percentile: Optional[float] = None,
    **_ignored,
    ) -> dict:

        """Capture ~duration_s seconds from current audio source and estimate:
        - noise_floor_colour (per-colour percentile)
        - noise_gate_db (RMS percentile + margin)
        """
        if self.audio_source is None:
            # Backward compatibility
            if self.stream is not None:
                self.audio_source = MicAudioSource(self.stream)
                self.audio_mode = "live"
            else:
                raise RuntimeError("No audio source open (call open_audio_source/open_microphone first).")

        n_frames = max(1, int(round(duration_s * self.sr / self.chunk_size)))
        rms_vals: list[float] = []
        col_vals: list[np.ndarray] = []

        buf = self.audio_buffer.copy()

        for _ in range(n_frames):
            x = self.audio_source.read(self.chunk_size)
            if x is None:
                break

            buf = np.roll(buf, -self.chunk_size)
            buf[-self.chunk_size:] = x

            rms_db = float(self._frame_rms_db(buf))
            rms_vals.append(rms_db)

            e = compute_canonical_energies(buf, self.cfg)
            c = (self.color_matrix @ e).astype(np.float32, copy=False)
            col_vals.append(c)

        if not rms_vals or not col_vals:
            raise RuntimeError("Calibration failed: no audio frames captured.")

        rms_p = float(np.percentile(np.asarray(rms_vals, dtype=np.float32), percentile))
        new_gate = float(rms_p + float(gate_margin_db))

        cols = np.stack(col_vals, axis=0)
        col_p = float(percentile if colour_percentile is None else colour_percentile)
        floor = np.percentile(cols, col_p, axis=0).astype(np.float32)


        self.set_noise_gate(new_gate)
        self.set_colour_noise_floor(floor)

        return {
            "duration_s": float(duration_s),
            "frames": int(len(rms_vals)),
            "percentile": float(percentile),
            "rms_percentile_db": rms_p,
            "noise_gate_db": new_gate,
            "noise_floor_colour": floor,
            "colour_percentile": col_p,
        }



    def set_bucket_freq(self, bucket_id: int, freq_index: int) -> None:
        """Set vibrotactile frequency index (0..7) for a given actuator/bucket."""
        if 0 <= bucket_id < self.n_actuators:
            self.freq_per_bucket[bucket_id] = int(freq_index) & 0x07

    def set_actuator_gain(self, actuator_id: int, gain: float) -> None:
        """Set per-actuator multiplicative gain (>=0)."""
        if 0 <= actuator_id < self.n_actuators:
            self.actuator_gains[actuator_id] = float(max(0.0, gain))

    def set_actuator_activation_range(self, actuator_id: int, min_duty: int, max_duty: int) -> None:
        """Set per-actuator duty gating/cap.

        min_duty:
            Below this -> output 0 (helps reduce buzzing)
        max_duty:
            Above this -> clamp to max
        """
        if 0 <= actuator_id < self.n_actuators:
            mn = int(np.clip(min_duty, 0, 15))
            mx = int(np.clip(max_duty, 0, 15))
            if mx < mn:
                mx = mn
            self.min_duty_per_act[actuator_id] = np.uint8(mn)
            self.max_duty_per_act[actuator_id] = np.uint8(mx)

    def set_all_bucket_edges(self, bucket_edges_hz: List[Tuple[float, float]], *, allow_overlap: bool = True) -> None:
        """Update spectral bucket boundaries (normal mode).

        This does NOT change the canonical front-end, only A.
        """
        if len(bucket_edges_hz) != self.cfg.M:
            raise ValueError(f"Expected {self.cfg.M} buckets, got {len(bucket_edges_hz)}")
        update_bucket_edges(self.cfg, bucket_edges_hz, allow_overlap=allow_overlap)

    def set_actuator_color(self, actuator_id: int, color_name: str) -> None:
        """Assign an actuator to a colour channel (colour mode)."""
        if 0 <= actuator_id < self.n_actuators:
            idx = self._color_name_to_idx.get(color_name)
            if idx is None:
                raise ValueError(f"Unknown colour: {color_name}")
            self.actuator_color_idx[actuator_id] = int(idx)

    # -----------------------------------------------------------------
    # BLE connect / disconnect
    # -----------------------------------------------------------------

    async def connect_vibraforge(self) -> bool:
        """Connect to the Vibraforge BLE control unit (robust).

        Fast path: try a list of known advertised names.
        Fallback: scan nearby devices, connect to candidates (strong RSSI first),
        and verify the expected characteristic UUID exists before accepting.
        """
        _ble_log("connect_vibraforge(): start")
        self.last_ble_scan = []
        self.last_ble_error = None
        self.ble_connected_name = None
        self.ble_connected_address = None
        self.ble_verify_ok = None  # True=verified, False=soft-verified/unknown

        
        async def _verify(client: BleakClient) -> Tuple[bool, List[str]]:
            """Try to locate CHARACTERISTIC_UUID in discovered services.

            Returns (ok, all_characteristic_uuids).
            On Windows/Bleak, service discovery can sometimes be flaky/slow right after connect,
            so we retry a few times with a short delay.
            """
            last_uuids: List[str] = []
            for attempt in range(3):
                try:
                    services = await client.get_services()
                    uuids: List[str] = []
                    for svc in services:
                        for ch in getattr(svc, "characteristics", []) or []:
                            try:
                                uuids.append(str(ch.uuid).lower())
                            except Exception:
                                pass
                    last_uuids = sorted(set(uuids))
                    if str(CHARACTERISTIC_UUID).lower() in set(last_uuids):
                        return True, last_uuids
                except Exception as e:
                    # keep info for debug
                    self.last_ble_error = f"get_services error: {e}"
                await asyncio.sleep(0.6)
            return False, last_uuids


        async def _connect_impl() -> Optional[BleakClient]:
            # 1) exact names first (quick)
            for nm in CONTROL_UNIT_NAME_CANDIDATES:
                try:
                    _ble_log(f"find_device_by_name: trying name={nm!r}")
                    dev = await BleakScanner.find_device_by_name(nm, timeout=3.0)
                except Exception as e:
                    self.last_ble_error = f"find_device_by_name error: {e}"
                    dev = None
                if not dev:
                    continue
                _ble_log(f"found device by name: name={getattr(dev,'name',None)} addr={getattr(dev,'address',None)}")
                try:
                    c = BleakClient(dev)
                    _ble_log(f"connect: attempting (name={getattr(dev,'name',None)} addr={getattr(dev,'address',None)})")
                    try:
                        await c.connect(timeout=10.0)
                    except TypeError:
                        await c.connect()
                    okv, uuids = await _verify(c)

                    if okv:
                        self.ble_connected_name = getattr(dev, "name", None)
                        self.ble_connected_address = getattr(dev, "address", None)
                        self.ble_verify_ok = True
                        _ble_log(f"verified characteristic ok; connected name={self.ble_connected_name} addr={self.ble_connected_address}")
                        return c
                    # Verification failed. In practice, Bleak service discovery can be flaky on Windows.
                    # If the device name strongly matches our expected firmware, proceed anyway (soft-verify).
                    dev_name = (getattr(dev, "name", "") or "").lower()
                    if any(tok in dev_name for tok in CONTROL_UNIT_NAME_TOKENS):
                        self.ble_connected_name = getattr(dev, "name", None)
                        self.ble_connected_address = getattr(dev, "address", None)
                        self.ble_verify_ok = False
                        self.last_ble_error = f"verify failed (char not enumerated). Proceeding with soft-verify. chars={uuids[:20]}"
                        _ble_log(f"verify failed; soft-verify accept name={self.ble_connected_name} addr={self.ble_connected_address}; chars={uuids[:20]}")
                        return c
                    _ble_log_msg = f"verify failed; disconnecting name={getattr(dev,'name',None)} addr={getattr(dev,'address',None)} chars={uuids[:20]}"
                    _ble_log(_ble_log_msg)
                    # Verification failed. Optionally proceed with soft-verify if the name matches.
                    dev_name = (getattr(dev, "name", "") or "").lower()
                    if any(tok in dev_name for tok in CONTROL_UNIT_NAME_TOKENS):
                        self.ble_connected_name = getattr(dev, "name", None)
                        self.ble_connected_address = getattr(dev, "address", None)
                        self.ble_verify_ok = False
                        self.last_ble_error = f"verify failed (char not enumerated). Proceeding with soft-verify. chars={uuids[:20]}"
                        _ble_log(f"verify failed; soft-verify accept name={self.ble_connected_name} addr={self.ble_connected_address}; chars={uuids[:20]}")
                        return c
                    _ble_log_msg = f"verify failed; disconnecting name={getattr(dev,'name',None)} addr={getattr(dev,'address',None)} chars={uuids[:20]}"
                    _ble_log(_ble_log_msg)
                    await c.disconnect()
                except Exception as e:
                    self.last_ble_error = f"connect/verify error (name={nm}): {e}"
                    try:
                        await c.disconnect()
                    except Exception:
                        pass

            # 2) scan + strongest signal first
            try:
                devices = await BleakScanner.discover(timeout=8.0)
                _ble_log(f"discover: found {len(devices)} devices")
                try:
                    for d in devices:
                        _ble_log(f"  - {getattr(d,'name',None)} [{getattr(d,'address',None)}] rssi={getattr(d,'rssi',None)}")
                except Exception:
                    pass
            except Exception as e:
                self.last_ble_error = f"discover error: {e}"
                devices = []

            self.last_ble_scan = [(getattr(d, "name", None), getattr(d, "address", "")) for d in devices]
            devices_sorted = sorted(devices, key=lambda d: getattr(d, "rssi", -999), reverse=True)

            # Prefer devices whose names look like our firmware; otherwise, try strongest RSSI.
            def _name_matches(d) -> bool:
                n = (getattr(d, "name", "") or "").lower()
                return any(tok in n for tok in CONTROL_UNIT_NAME_TOKENS)

            named = [d for d in devices_sorted if _name_matches(d)]
            candidates = named if named else list(devices_sorted)

            for dev in candidates[:20]:
                try:
                    _ble_log(f"candidate connect: name={getattr(dev,'name',None)} addr={getattr(dev,'address',None)} rssi={getattr(dev,'rssi',None)}")
                    c = BleakClient(dev)
                    _ble_log(f"connect: attempting (name={getattr(dev,'name',None)} addr={getattr(dev,'address',None)})")
                    try:
                        await c.connect(timeout=10.0)
                    except TypeError:
                        await c.connect()
                    okv, uuids = await _verify(c)

                    if okv:
                        self.ble_connected_name = getattr(dev, "name", None)
                        self.ble_connected_address = getattr(dev, "address", None)
                        self.ble_verify_ok = True
                        _ble_log(f"verified characteristic ok; connected name={self.ble_connected_name} addr={self.ble_connected_address}")
                        return c
                    await c.disconnect()
                except Exception as e:
                    self.last_ble_error = f"connect/verify error (addr={getattr(dev,'address',None)}): {e}"
                    try:
                        await c.disconnect()
                    except Exception:
                        pass

            return None

        try:
            if sys.platform == "win32":
                self._ble_thread.start()  # type: ignore[union-attr]
                async def _wrap():
                    return await _connect_impl()
                self.client = await self._ble_thread.run(_wrap())  # type: ignore[union-attr]
            else:
                self.client = await _connect_impl()
        except Exception as e:
            self.last_ble_error = str(e)
            self.client = None

        self.is_connected = self.client is not None
        _ble_log(f"connect_vibraforge(): done -> is_connected={self.is_connected} name={getattr(self,'ble_connected_name',None)} addr={getattr(self,'ble_connected_address',None)} verify_ok={getattr(self,'ble_verify_ok',None)} err={getattr(self,'last_ble_error',None)}")
        return self.is_connected

    async def disconnect_vibraforge(self) -> None:
        """Disconnect BLE client (best effort)."""
        try:
            if not (self.is_connected and self.client):
                return

            if sys.platform == "win32":
                async def _disc():
                    await self.client.disconnect()
                await self._ble_thread.run(_disc())  # type: ignore[union-attr]
            else:
                await self.client.disconnect()
        finally:
            self.client = None
            self.is_connected = False
            if sys.platform == "win32" and self._ble_thread:
                self._ble_thread.stop()

    # -----------------------------------------------------------------
    # Vibraforge packet helpers
    # -----------------------------------------------------------------

    def create_command(self, addr: int, mode: int, duty: int, freq: int) -> bytearray:
        """Create a 3-byte Vibraforge command (matches vibraforge_test.py)."""
        serial_group = addr // 16
        serial_addr = addr % 16
        byte1 = (serial_group << 2) | (mode & 0x01)
        byte2 = 0x40 | (serial_addr & 0x3F)
        byte3 = 0x80 | ((duty & 0x0F) << 3) | (freq & 0x07)
        return bytearray([byte1, byte2, byte3])

    @staticmethod
    def _pad_packet(packet: bytearray) -> bytearray:
        while len(packet) < PACKET_SIZE_BYTES:
            packet.extend([0xFF, 0xFF, 0xFF])
        return packet[:PACKET_SIZE_BYTES]

    async def _write_packet(self, packet: bytearray) -> None:
        if not (self.is_connected and self.client):
            return

        if sys.platform == "win32":
            async def _write():
                await self.client.write_gatt_char(CHARACTERISTIC_UUID, packet)
            await self._ble_thread.run(_write())  # type: ignore[union-attr]
            return

        await self.client.write_gatt_char(CHARACTERISTIC_UUID, packet)

    def _desired_duty_per_actuator(self, intensity_levels: np.ndarray) -> Dict[int, int]:
        """Combine multiple channel contributions safely.

        If multiple channels map to the same actuator address, we take MAX duty.
        (non-additive)
        """
        out: Dict[int, int] = {}
        for m in self.mapping:
            if m.bucket_id >= len(intensity_levels):
                continue
            duty = int(intensity_levels[m.bucket_id])
            for addr in m.actuator_ids:
                out[addr] = max(out.get(addr, 0), duty)
        return out

    # -----------------------------------------------------------------
    # High-level haptic control
    # -----------------------------------------------------------------

    async def send_haptic_command(self, intensity_levels: np.ndarray) -> None:
        """Send commands for mapped actuators.

        Non-additive reset behaviour:
        - duty==0 -> STOP
        - if duty/freq changed while running, optionally STOP(old) then START(new)
        """
        if not self.is_connected or not self.client:
            return

        desired = self._desired_duty_per_actuator(intensity_levels)
        cmds: List[bytearray] = []

        for addr in self.mapped_actuators:
            new_duty = int(desired.get(addr, 0))

            bucket_id_for_act = self.actuator_to_bucket.get(addr, 0)
            new_freq = self.freq_per_bucket[bucket_id_for_act] if 0 <= bucket_id_for_act < len(self.freq_per_bucket) else self.freq_index

            new_mode = 1 if new_duty > 0 else 0
            old_mode, old_freq, old_duty = self._last_state.get(addr, (0, new_freq, 0))

            need_stop = False
            if new_mode == 0:
                need_stop = (old_mode == 1 and old_duty > 0)
            else:
                if self.reset_on_change and (old_mode == 1 and old_duty > 0):
                    if old_freq != new_freq or old_duty != new_duty:
                        need_stop = True

            if need_stop:
                cmds.append(self.create_command(addr, mode=0, duty=0, freq=old_freq))

            if new_mode == 1:
                cmds.append(self.create_command(addr, mode=1, duty=new_duty, freq=new_freq))

            self._last_state[addr] = (new_mode, new_freq, new_duty if new_mode == 1 else 0)

        for i in range(0, len(cmds), MAX_COMMANDS_PER_PACKET):
            packet = bytearray().join(cmds[i: i + MAX_COMMANDS_PER_PACKET])
            await self._write_packet(self._pad_packet(packet))

        self._last_ok_time = time.monotonic()

    async def stop_mapped_actuators(self) -> None:
        """Stop ONLY mapped actuators (best effort), chunked safely."""
        if not self.is_connected or not self.client:
            return

        cmds: List[bytearray] = []
        for addr in self.mapped_actuators:
            old_mode, old_freq, old_duty = self._last_state.get(addr, (0, self.freq_index, 0))
            if old_mode == 1 and old_duty > 0:
                cmds.append(self.create_command(addr, mode=0, duty=0, freq=old_freq))
            self._last_state[addr] = (0, old_freq, 0)

        for i in range(0, len(cmds), MAX_COMMANDS_PER_PACKET):
            packet = bytearray().join(cmds[i: i + MAX_COMMANDS_PER_PACKET])
            await self._write_packet(self._pad_packet(packet))

    async def panic_stop(self) -> None:
        """Best-effort stop, safe to call from finally blocks and GUI hotkey."""
        try:
            await self.stop_mapped_actuators()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Audio input
    # -----------------------------------------------------------------


    def _find_best_input_device_index(self, name_contains: str, preferred_sr: int) -> Optional[int]:
        """Return the input device index whose name contains text and whose default SR is closest to preferred_sr."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        needle = name_contains.lower().strip()
        best = None
        best_score = 1e18

        for i in range(self._pyaudio.get_device_count()):
            d = self._pyaudio.get_device_info_by_index(i)
            if d.get("maxInputChannels", 0) <= 0:
                continue
            name = str(d.get("name", ""))
            if needle not in name.lower():
                continue

            sr = float(d.get("defaultSampleRate", 0.0))
            score = abs(sr - float(preferred_sr))
            if score < best_score:
                best_score = score
                best = i

        return best

    def open_microphone(self, device_name_contains: Optional[str] = None) -> None:
        """Open the microphone stream. If device_name_contains is set, pick a matching input device."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        # Close any stale stream
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        device_index = None
        if device_name_contains:
            device_index = self._find_best_input_device_index(device_name_contains, preferred_sr=self.sr)
            print(f"[Audio] Requested '{device_name_contains}' -> selected index: {device_index}")

        # Open stream (None index = PortAudio default input)
        self.stream = self._pyaudio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
        )
        # Wrap into audio source when open_microphone() is used directly
        self.audio_source = MicAudioSource(self.stream)
        self.audio_mode = "live"

    def close_microphone(self) -> None:
        """Close microphone stream (and any wrapped audio source) without terminating PyAudio."""
        if self.audio_source is not None and isinstance(self.audio_source, (MicAudioSource, RecordingAudioSource)):
            try:
                self.audio_source.close()
            except Exception:
                pass
            self.audio_source = None

        if self.stream is None:
            return
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.stream = None

    def close_audio_source(self) -> None:
        """Close whichever audio source is active (mic/record/playback)."""
        if self.audio_source is not None:
            try:
                self.audio_source.close()
            except Exception:
                pass
            self.audio_source = None

        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def open_audio_source(
        self,
        mode: str,
        *,
        input_device_query: Optional[str] = None,
        record_path: Optional[str] = None,
        playback_path: Optional[str] = None,
        playback_to_speaker: bool = False,
    ) -> None:
        """Select audio source:
        - mode='live': microphone
        - mode='record': microphone + write to wav
        - mode='replay': play a wav file (optionally to speaker) + drive haptics
        """
        mode_n = mode.strip().lower()
        if mode_n == "playback":
            mode_n = "replay"
        if mode_n not in {"live", "record", "replay"}:
            raise ValueError(f"Unknown audio mode: {mode}")

        self.close_audio_source()
        self.audio_mode = mode_n

        if mode_n in {"live", "record"}:
            self.open_microphone(device_name_contains=input_device_query)
            if self.stream is None:
                raise RuntimeError("Failed to open microphone stream.")
            mic = MicAudioSource(self.stream)
            if mode_n == "record":
                if not record_path:
                    raise ValueError("record_path is required for record mode.")
                try:
                    os.makedirs(os.path.dirname(str(record_path)) or ".", exist_ok=True)
                except Exception:
                    pass
                self.audio_source = RecordingAudioSource(mic, record_path=str(record_path), sr=self.sr)
            else:
                self.audio_source = mic
            return

        if not playback_path:
            raise ValueError("playback_path is required for replay mode.")
        if not os.path.exists(str(playback_path)):
            raise FileNotFoundError(f"Playback file not found: {playback_path}")
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        self.audio_source = PlaybackAudioSource(
            pya=self._pyaudio,
            playback_path=str(playback_path),
            target_sr=self.sr,
            chunk_size=self.chunk_size,
            to_speaker=bool(playback_to_speaker),
        )


    def terminate_audio_backend(self) -> None:
        """Terminate PyAudio (call during final shutdown)."""
        if self._pyaudio is None:
            return
        try:
            self._pyaudio.terminate()
        except Exception:
            pass
        finally:
            self._pyaudio = None

    @staticmethod
    def _frame_rms_db(x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean((x.astype(np.float32) ** 2)) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12)

    def _update_rms_history(self, rms_db: float) -> None:
        now = time.monotonic()
        self.rms_history.append((now, float(rms_db)))
        cutoff = now - self.max_history_s
        while self.rms_history and self.rms_history[0][0] < cutoff:
            self.rms_history.pop(0)

    def get_rms_stats(self):
        """Return (mean5, peak5, mean10, peak10) in dB (NaN if no data)."""
        now = time.monotonic()
        vals5 = [v for t, v in self.rms_history if now - t <= 5.0]
        vals10 = [v for t, v in self.rms_history if now - t <= 10.0]

        def stats(vals):
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(max(vals))

        mean5, peak5 = stats(vals5)
        mean10, peak10 = stats(vals10)
        return mean5, peak5, mean10, peak10

    # -----------------------------------------------------------------
    # Audio -> channel intensities
    # -----------------------------------------------------------------

    def _apply_actuator_postprocessing(self, duties: np.ndarray) -> np.ndarray:
        """Apply per-actuator gain and activation min/max to duties."""
        duties_f = duties.astype(np.float32)
        n = min(len(duties_f), len(self.actuator_gains))
        duties_f[:n] *= self.actuator_gains[:n]
        duties_f = np.clip(np.round(duties_f), 0, 15)
        duties_u8 = duties_f.astype(np.uint8)

        # Apply min/max duty constraints per actuator
        out = duties_u8.copy()
        for i in range(min(len(out), self.n_actuators)):
            mn = int(self.min_duty_per_act[i])
            mx = int(self.max_duty_per_act[i])
            v = int(out[i])
            if v < mn:
                out[i] = 0
            elif v > mx:
                out[i] = np.uint8(mx)
        return out
    def _spectral_flatness(self, frame: np.ndarray, f_lo: float = 300.0, f_hi: float = 8000.0) -> float:
        """Compute spectral flatness (0..1) on a frequency band.

        Lower values typically correspond to harmonic/voiced content.
        Higher values typically correspond to noise-like content (breath, hiss, fans).
        """
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size < 8:
            return float("nan")
        w = np.hanning(x.size).astype(np.float32)
        X = np.fft.rfft(x * w)
        mag = np.abs(X).astype(np.float32) + 1e-12
        freqs = np.fft.rfftfreq(x.size, d=1.0 / float(self.sr))
        msk = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
        if not np.any(msk):
            return float("nan")
        v = mag[msk]
        gm = float(np.exp(np.mean(np.log(v))))
        am = float(np.mean(v))
        return float(gm / (am + 1e-12))

    def _voicedness_from_flatness(self, flatness: float) -> float:
        """Map flatness -> voicedness factor in [voicedness_min, 1]."""
        if not np.isfinite(flatness):
            return 1.0
        lo = float(self.voiced_flatness_floor)
        hi = float(self.voiced_flatness_thresh)
        vmin = float(self.voicedness_min)
        if flatness <= lo:
            return 1.0
        if flatness >= hi:
            return vmin
        t = (flatness - lo) / max(hi - lo, 1e-6)
        return float((1.0 - t) * 1.0 + t * vmin)


    def process_audio_frame(self) -> Optional[np.ndarray]:
        """Read one chunk, update rolling FFT frame, return actuator duties.

        Returns
        -------
        duties : np.ndarray (uint8), length = n_actuators
            Duty per actuator (0..15).
        """
        if self.audio_source is None:
            if self.stream is not None:
                self.audio_source = MicAudioSource(self.stream)
                self.audio_mode = "live"
            else:
                return None

        try:
            audio_chunk = self.audio_source.read(self.chunk_size)
        except Exception as e:
            print(f"WARNING: Audio source error: {e}")
            self.close_audio_source()
            return None

        if audio_chunk is None:
            # End-of-stream (replay finished) or mic error
            self.close_audio_source()
            return None

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        self.audio_buffer = np.roll(self.audio_buffer, -self.chunk_size)
        self.audio_buffer[-self.chunk_size:] = audio_chunk

        rms_db = self._frame_rms_db(self.audio_buffer)
        self.last_rms_db = float(rms_db)
        self._update_rms_history(rms_db)


        # Voicedness estimate (helps reduce activation on breath / broadband noise).
        voicedness = 1.0
        if self.voicedness_gate_enabled:
            flat = self._spectral_flatness(self.audio_buffer)
            self.last_spectral_flatness = float(flat) if np.isfinite(flat) else float("nan")
            voicedness = self._voicedness_from_flatness(self.last_spectral_flatness)
            self.last_voicedness = float(voicedness)
        # Silence -> immediately output zero duties
        if rms_db < self.noise_gate_db:
            return np.zeros(self.n_actuators, dtype=np.uint8)

        # ---------------- Normal mode: M bucket energies -> M duties ----------------
        if self.mode == "normal":
            bucket_energies = compute_bucket_energies(self.audio_buffer, self.cfg)
            if self.voicedness_gate_enabled:
                bucket_energies = bucket_energies * float(voicedness)
            self.last_bucket_energies = bucket_energies.astype(np.float32, copy=False)

            # Optional mild bias favoring lows (feel stronger tactually).
            # You can tune or disable by setting bias_db=None.
            bias_db = None  #np.linspace(0.0, -10.0, num=len(bucket_energies), dtype=np.float32)

            duties = energies_to_intensity(
                bucket_energies,
                rms_db=rms_db,
                noise_gate_db=self.noise_gate_db,
                max_duty=12,
                gamma=float(self.loudness_gamma),
                bias_db=bias_db,
                spectral_span_db=float(self.spectral_span_db),
            )
            return self._apply_actuator_postprocessing(duties)

        # ---------------- Colour mode: K canonical -> C colours -> N actuators ----------------
        # 1) canonical energies (stable)
        e = compute_canonical_energies(self.audio_buffer, self.cfg)  # (K,)
        self.last_canonical_energies = e.astype(np.float32, copy=False)

        # 2) colour energies
        c_raw = (self.color_matrix @ e).astype(np.float32, copy=False)  # (C,)
        if self.voicedness_gate_enabled:
            c_raw = c_raw * float(voicedness)
        self.last_colour_energies_raw = c_raw

        # 2b) per-colour noise floor subtraction (optional)
        c = c_raw
        if self.colour_noise_subtract_enabled and (self.noise_floor_colour is not None):
            floor = np.asarray(self.noise_floor_colour, dtype=np.float32).reshape(-1)
            if floor.shape[0] == c_raw.shape[0]:
                c = np.maximum(0.0, c_raw - (self.colour_noise_subtract_alpha * floor))
        self.last_colour_energies_clean = c.astype(np.float32, copy=False)

        # Optional dominance (grouped, non-destructive)
        c_drive = self._apply_colour_dominance(c)
        self.last_colour_energies_drive = c_drive.astype(np.float32, copy=False)

        # 3) colour duties
        colour_duties = energies_to_intensity(
            c_drive,
            rms_db=rms_db,
            noise_gate_db=self.noise_gate_db,
            max_duty=12,
            gamma=float(self.loudness_gamma),
            bias_db=None,
            spectral_span_db=float(self.spectral_span_db),
        )

        # 4) map colour duties -> actuator duties
        out = np.zeros(self.n_actuators, dtype=np.uint8)
        for a in range(self.n_actuators):
            ci = int(self.actuator_color_idx[a])
            if 0 <= ci < len(colour_duties):
                out[a] = np.uint8(colour_duties[ci])

        return self._apply_actuator_postprocessing(out)

    # -----------------------------------------------------------------
    # Shutdown helper (important for Stop/Start reliability)
    # -----------------------------------------------------------------

    async def shutdown(self) -> None:
        """Fully release hardware resources (best effort)."""
        # Close audio first
        self.close_audio_source()
        try:
            await self.panic_stop()
        except Exception:
            pass

        try:
            self.close_microphone()
        except Exception:
            pass

        try:
            await self.disconnect_vibraforge()
        except Exception:
            pass

        # Terminate PyAudio last
        try:
            self.terminate_audio_backend()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Optional: self-contained realtime loop
    # -----------------------------------------------------------------

    async def run_realtime_loop(self, duration_seconds: Optional[float] = None) -> None:
        """Standalone loop (useful for CLI tests).

        The GUI uses process_audio_frame() directly, so this is optional.
        """
        self.open_microphone()
        self._last_ok_time = time.monotonic()
        start = time.monotonic()

        try:
            while True:
                if duration_seconds is not None and (time.monotonic() - start) > duration_seconds:
                    break

                duties = self.process_audio_frame()
                if duties is None:
                    if (time.monotonic() - self._last_ok_time) > self.idle_stop_timeout_s:
                        await self.panic_stop()
                        self._last_ok_time = time.monotonic()
                    await asyncio.sleep(0.01)
                    continue

                await self.send_haptic_command(duties)
                await asyncio.sleep(0.001)

        finally:
            await self.shutdown()