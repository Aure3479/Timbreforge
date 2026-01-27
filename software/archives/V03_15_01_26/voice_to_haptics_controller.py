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
# BLE constants – change if your Vibraforge firmware changes
# -----------------------------------------------------------------------------

CHARACTERISTIC_UUID = "f22535de-5375-44bd-8ca9-d0ea9ff9e410"
CONTROL_UNIT_NAME = "QT Py ESP32-S3"

MAX_COMMANDS_PER_PACKET = 20
PACKET_SIZE_BYTES = 60


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
    ch_db = 20.0 * np.log10(e)
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

    duties = np.round((g * s) * float(max_duty)).astype(np.uint8)
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
        
    
        # --------- Actuator routing (baseline identity mapping) ---------
        self.mapping = default_actuator_mapping_identity(self.n_actuators)
        self.mapped_actuators = sorted({a for m in self.mapping for a in m.actuator_ids})

        # Each actuator belongs to exactly one "bucket" in baseline identity mapping.
        # We use this for frequency selection.
        self.actuator_to_bucket: Dict[int, int] = {}
        for m in self.mapping:
            for addr in m.actuator_ids:
                self.actuator_to_bucket[addr] = m.bucket_id

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

    def set_mode(self, mode: str) -> None:
        """Set analysis mode: 'normal' or 'color'."""
        mode = mode.strip().lower()
        if mode not in {"normal", "color"}:
            raise ValueError("mode must be 'normal' or 'color'")
        self.mode = mode

    def set_noise_gate(self, noise_gate_db: float) -> None:
        """Set silence threshold in dB."""
        self.noise_gate_db = float(noise_gate_db)

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
        """Scan by name and connect."""
        if sys.platform == "win32":
            self._ble_thread.start()  # type: ignore[union-attr]

            async def _connect():
                device = await BleakScanner.find_device_by_name(CONTROL_UNIT_NAME, timeout=6.0)
                if not device:
                    return None
                client = BleakClient(device)
                await client.connect()
                return client

            self.client = await self._ble_thread.run(_connect())  # type: ignore[union-attr]
            self.is_connected = self.client is not None
            return self.is_connected

        device = await BleakScanner.find_device_by_name(CONTROL_UNIT_NAME, timeout=6.0)
        if not device:
            return False
        self.client = BleakClient(device)
        await self.client.connect()
        self.is_connected = True
        return True

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

    def close_microphone(self) -> None:
        """Close the microphone stream (does NOT terminate PyAudio)."""
        if self.stream is None:
            return
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.stream = None

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

    def process_audio_frame(self) -> Optional[np.ndarray]:
        """Read one chunk, update rolling FFT frame, return actuator duties.

        Returns
        -------
        duties : np.ndarray (uint8), length = n_actuators
            Duty per actuator (0..15).
        """
        if self.stream is None:
            return None

        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
        except OSError as e:
        # ========== FIX #3: Clean close on audio device error ==========
        # This prevents spinning loop if mic is unplugged mid-stream
            print(f"WARNING: Audio stream error: {e} (closing microphone)")
            self.close_microphone()  # Clean close
            return None  # Next frame will handle None gracefully

        audio_chunk = np.frombuffer(data, dtype=np.float32)
        self.audio_buffer = np.roll(self.audio_buffer, -self.chunk_size)
        self.audio_buffer[-self.chunk_size:] = audio_chunk

        rms_db = self._frame_rms_db(self.audio_buffer)
        self._update_rms_history(rms_db)

        # Silence -> immediately output zero duties
        if rms_db < self.noise_gate_db:
            return np.zeros(self.n_actuators, dtype=np.uint8)

        # ---------------- Normal mode: M bucket energies -> M duties ----------------
        if self.mode == "normal":
            bucket_energies = compute_bucket_energies(self.audio_buffer, self.cfg)

            # Optional mild bias favoring lows (feel stronger tactually).
            # You can tune or disable by setting bias_db=None.
            bias_db = None  #np.linspace(0.0, -10.0, num=len(bucket_energies), dtype=np.float32)

            duties = energies_to_intensity(
                bucket_energies,
                rms_db=rms_db,
                noise_gate_db=self.noise_gate_db,
                max_duty=12,
                bias_db=bias_db,
            )
            return self._apply_actuator_postprocessing(duties)

        # ---------------- Colour mode: K canonical -> C colours -> N actuators ----------------
        # 1) canonical energies (stable)
        e = compute_canonical_energies(self.audio_buffer, self.cfg)  # (K,)
        # 2) colour energies
        c = self.color_matrix @ e  # (C,)
        # 3) colour duties
        colour_duties = energies_to_intensity(
            c,
            rms_db=rms_db,
            noise_gate_db=self.noise_gate_db,
            max_duty=12,
            bias_db=None,
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
