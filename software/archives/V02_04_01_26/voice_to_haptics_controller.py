# voice_to_haptics_controller.py
"""
================================================================================
Vibraforge – Voice → Haptics Controller (Realtime)
================================================================================

WHAT THIS FILE DOES
- Captures microphone audio (PyAudio)
- Extracts M bucket energies (bucket_classification.py)
- Converts bucket energies → intensities (0..15)
- Maps intensities → actuator addresses
- Sends Vibraforge commands over BLE (Bleak) to the ESP32 controller

BASELINE SETUP (YOUR CURRENT CHOICE)
- We use 6 buckets and we map them to actuator addresses 0..5 (first six motors).
  This is the simplest sanity-check configuration.

THE 3 MOST IMPORTANT SETTINGS YOU MAY NEED TO CHANGE
1) BLE device name / UUID
   - CONTROL_UNIT_NAME must match your Vibraforge device advertising name
   - CHARACTERISTIC_UUID must be the writable characteristic

2) Mapping (bucket → actuator addresses)
   - default_actuator_mapping_m6(): choose which physical motors correspond to
     bucket 0..5. For baseline: addresses 0..5.

3) Audio sensitivity (if motors buzz on silence)
   - Look for NOISE_GATE_DB or intensity conversion settings.
   - You can increase the gate (e.g. -45 dB) to require louder voice.

SAFETY FEATURES INCLUDED
✅ Panic stop method (panic_stop) for emergency shutdown
✅ Dead-man failsafe: if audio processing stalls, it stops motors
✅ Non-additive reset:
   - If freq/duty changes while running, we send STOP(old) then START(new)
   - This prevents “stacking” if firmware layers commands

COMMAND PROTOCOL NOTES
- Packet is 60 bytes = 20 commands × 3 bytes each.
- If we need more than 20 commands, we send multiple packets.

================================================================================
"""
import sys
import threading
import concurrent.futures


import asyncio
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from bleak import BleakScanner, BleakClient
import pyaudio

from bucket_classification import compute_bucket_energies, make_bucket_config


CHARACTERISTIC_UUID = "f22535de-5375-44bd-8ca9-d0ea9ff9e410"
CONTROL_UNIT_NAME = "QT Py ESP32-S3"

MAX_COMMANDS_PER_PACKET = 20
PACKET_SIZE_BYTES = 60


class _BleakWinMtaThread:
    """
    =================================================================================
    WINDOWS BLEAK FIX (STA → MTA)
    =================================================================================
    Why this exists:
    - On Windows, Bleak’s WinRT backend requires running in an MTA thread for console apps.
    - If the main thread ends up STA, Bleak scanning/connect will fail with:
        "Thread is configured for Windows GUI but callbacks are not working."
      (No Windows message loop => WinRT callbacks never complete.) :contentReference[oaicite:1]{index=1}

    What this class does:
    - Starts a dedicated background thread
    - Creates an asyncio event loop inside that thread
    - Runs ALL Bleak operations (scan/connect/write/disconnect) on that loop

    When to modify:
    - Almost never. If you remove it, you risk the STA/MTA crash coming back.
    =================================================================================
    """
    def __init__(self):
        self.loop = None
        self.thread = None

    def start(self):
        if self.loop is not None:
            return
        self.loop = asyncio.new_event_loop()

        def _runner():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(target=_runner, daemon=True)
        self.thread.start()

    async def run(self, coro):
        # schedule a coroutine onto the BLE thread loop and await its result
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return await asyncio.wrap_future(fut)

    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop = None
            self.thread = None


@dataclass
class ActuatorMapping:
    bucket_id: int
    actuator_ids: List[int]
    body_location: str = ""


def default_actuator_mapping_m6() -> List[ActuatorMapping]:
    """
    BASELINE: bucket 0..5 → actuator address 0..5.

    If later you discover your physical motors respond on different addresses,
    this is the ONLY place you need to edit for routing.
    """
    return [
        ActuatorMapping(bucket_id=0, actuator_ids=[0], body_location="A0"),
        ActuatorMapping(bucket_id=1, actuator_ids=[1], body_location="A1"),
        ActuatorMapping(bucket_id=2, actuator_ids=[2], body_location="A2"),
        ActuatorMapping(bucket_id=3, actuator_ids=[3], body_location="A3"),
        ActuatorMapping(bucket_id=4, actuator_ids=[4], body_location="A4"),
        ActuatorMapping(bucket_id=5, actuator_ids=[5], body_location="A5"),
    ]


def bucket_energies_to_intensity(
    bucket_energies: np.ndarray,
    rms_db: float,
    noise_gate_db: float,
    max_duty: int = 10,
    loudness_span_db: float = 20.0,
    spectral_span_db: float = 25.0,
) -> np.ndarray:
    """
    Convert bucket energies -> duty 0..max_duty using:
    - global loudness (RMS) to scale overall strength
    - relative spectral shape to decide which buckets are dominant

    Parameters
    ----------
    bucket_energies : array-like, shape (n_buckets,)
        Positive energy per bucket.
    rms_db : float
        RMS dB of the full frame (used for loudness factor).
    noise_gate_db : float
        Gate: if rms_db < noise_gate_db, intensities are all 0.
    max_duty : int
        Maximum duty we ever send (<= 15).
    loudness_span_db : float
        How many dB above the gate we need to go from 0 → full loudness.
    spectral_span_db : float
        Dynamic range within the spectrum. Buckets more than this below the
        loudest bucket are treated as "off" (0).

    Intuition
    ---------
    - If you speak just above the gate, everything is soft.
    - If you speak much louder, everything scales up.
    - In each frame, only buckets close to the loudest one get near max duty;
      others get proportionally less.
    """
    eps = 1e-10
    e = np.maximum(bucket_energies, 0.0) + eps

    # --- 0. Noise gate: silence -> all zeros ---
    if rms_db < noise_gate_db:
        return np.zeros_like(e, dtype=np.uint8)

    # --- 1. Global loudness factor (0..1) based on RMS ---
    # Map [noise_gate_db, noise_gate_db + loudness_span_db] -> [0, 1]
    g = (rms_db - noise_gate_db) / float(loudness_span_db)
    g = float(np.clip(g, 0.0, 1.0))

    # --- 2. Spectral contrast per frame ---
    bucket_db = 20.0 * np.log10(e)
    max_db = float(bucket_db.max())

    # If something goes very wrong:
    if not np.isfinite(max_db):
        return np.zeros_like(e, dtype=np.uint8)

    # Optional bias: emphasize low frequencies, attenuate highs
    # (feel free to tweak these offsets or remove them)
    bucket_bias_db = np.array([0.0, 0.0, -3.0, -6.0, -9.0, -12.0][: len(bucket_db)])
    bucket_db = bucket_db + bucket_bias_db

    # Define a spectral floor: anything more than spectral_span_db below max is "off"
    floor = max_db - spectral_span_db
    # Normalize: buckets at floor -> 0, at max -> 1
    s = (bucket_db - floor) / (max_db - floor + 1e-6)
    s[bucket_db <= floor] = 0.0
    s = np.clip(s, 0.0, 1.0)

    # --- 3. Combine global loudness (g) and spectral shape (s) ---
    vals = g * s  # still in [0, 1]
    intensities = np.round(vals * max_duty).astype(np.uint8)

    return intensities


class VoiceToHapticsController:
    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        chunk_size: int = 1024,
        n_buckets: int = 6,
        bucket_method: str = "voice_landmark",
        freq_index: int = 5,                 # Vibraforge frequency index (0..7)
        reset_on_change: bool = True,        # STOP(old) → START(new) if duty/freq changes
        idle_stop_timeout_s: float = 0.30,   # dead-man: stop motors if loop stalls
        noise_gate_db: float = -60.0,        # raise to -50 or -45 if you get buzzing on silence
    ):
        self._ble_thread = _BleakWinMtaThread() if sys.platform == "win32" else None
        self.sr = sr
        self.n_fft = n_fft
        self.chunk_size = chunk_size
        self.n_buckets = n_buckets
        self.bucket_method = bucket_method

        self.freq_index = int(freq_index) & 0x07
        self.reset_on_change = bool(reset_on_change)
        self.idle_stop_timeout_s = float(idle_stop_timeout_s)
        self.noise_gate_db = float(noise_gate_db)

        self.cfg = make_bucket_config(
            sr=sr,
            n_fft=n_fft,
            fmin=50.0,
            fmax=8000.0,
            K=32,
            M=n_buckets,
            bucket_method=bucket_method,
            inner_scale="erb",
        )

        self.mapping = default_actuator_mapping_m6() if n_buckets == 6 else []
        self.mapped_actuators = sorted({a for m in self.mapping for a in m.actuator_ids})
                
                
                # Per-bucket gains (live-tunable from GUI)
        # - 1.0  => neutral
        # - <1.0 => make this bucket weaker
        # - >1.0 => make this bucket stronger
        self.bucket_gains = np.ones(self.n_buckets, dtype=np.float32)

                # Map each actuator address back to its "bucket owner".
        # For now each actuator appears at most once in mapping (baseline 0..5).
        self.actuator_to_bucket: Dict[int, int] = {}
        for m in self.mapping:
            for addr in m.actuator_ids:
                self.actuator_to_bucket[addr] = m.bucket_id

        # Per-bucket frequencies (0..7).
        # By default, every bucket uses the global freq_index you pass to __init__.
        self.freq_per_bucket = [self.freq_index] * self.n_buckets

        # RMS / loudness history for GUI:
        # we keep (timestamp, rms_db) for the last ~10 seconds.
        self.rms_history: list[tuple[float, float]] = []
        self.max_history_s: float = 10.0


        self._last_state: Dict[int, Tuple[int, int, int]] = {}  # addr -> (mode, freq, duty)
        self._last_ok_time = time.monotonic()

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = np.zeros(n_fft, dtype=np.float32)

        self.client: Optional[BleakClient] = None
        self.is_connected = False

    # ---------------- BLE ----------------

    async def connect_vibraforge(self) -> bool:
        # Always do Bleak work on the MTA thread on Windows.
        if sys.platform == "win32":
            self._ble_thread.start()

            async def _connect():
                device = await BleakScanner.find_device_by_name(CONTROL_UNIT_NAME, timeout=6.0)
                if not device:
                    return None
                client = BleakClient(device)
                await client.connect()
                return client

            self.client = await self._ble_thread.run(_connect())
            self.is_connected = self.client is not None
            return self.is_connected

        
        # Non-Windows: normal path
        device = await BleakScanner.find_device_by_name(CONTROL_UNIT_NAME, timeout=6.0)
        if not device:
            return False
        self.client = BleakClient(device)
        await self.client.connect()
        self.is_connected = True
        return True

    def create_command(self, addr: int, mode: int, duty: int, freq: int) -> bytearray:
        """
        Vibraforge command format (matches your working vibraforge_test.py).
        """
        serial_group = addr // 16
        serial_addr = addr % 16
        byte1 = (serial_group << 2) | (mode & 0x01)
        byte2 = 0x40 | (serial_addr & 0x3F)
        byte3 = 0x80 | ((duty & 0x0F) << 3) | (freq & 0x07)
        return bytearray([byte1, byte2, byte3])

    def _pad_packet(self, packet: bytearray) -> bytearray:
        while len(packet) < PACKET_SIZE_BYTES:
            packet.extend([0xFF, 0xFF, 0xFF])
        return packet[:PACKET_SIZE_BYTES]

    async def _write_packet(self, packet: bytearray):
        if not (self.is_connected and self.client):
            return

        if sys.platform == "win32":
            async def _write():
                await self.client.write_gatt_char(CHARACTERISTIC_UUID, packet)
            await self._ble_thread.run(_write())
            return

        await self.client.write_gatt_char(CHARACTERISTIC_UUID, packet)

    def _desired_duty_per_actuator(self, intensity_levels: np.ndarray) -> Dict[int, int]:
        """
        Combine multiple bucket contributions safely:
        - If multiple buckets map to same actuator, we take MAX duty (non-additive).
        """
        out: Dict[int, int] = {}
        for m in self.mapping:
            if m.bucket_id >= len(intensity_levels):
                continue
            duty = int(intensity_levels[m.bucket_id])
            for addr in m.actuator_ids:
                out[addr] = max(out.get(addr, 0), duty)
        return out
        # --------- Configuration from GUI ---------

    def set_bucket_freq(self, bucket_id: int, freq_index: int) -> None:
        """
        Set the frequency index (0..7) used for a given bucket.

        NOTE:
        - In baseline mapping, bucket i -> actuator i, so this directly changes
          the frequency of actuator i.
        - If an actuator belongs to multiple buckets (future layouts),
          we still use the bucket->actuator mapping to decide frequency.
        """
        if 0 <= bucket_id < self.n_buckets:
            self.freq_per_bucket[bucket_id] = int(freq_index) & 0x07

    def set_noise_gate(self, noise_gate_db: float) -> None:
        """
        Change the global noise gate (in dB).

        If the RMS dB of the full audio frame is below this value, we treat
        it as silence and send all-zero intensities (no vibration).
        """
        self.noise_gate_db = float(noise_gate_db)

    def set_bucket_gain(self, bucket_id: int, gain: float) -> None:
        """
        Set a multiplicative gain for a given bucket.

        Parameters
        ----------
        bucket_id : int
            Which bucket (0..n_buckets-1) to adjust.
        gain : float
            - 1.0  => neutral (no change)
            - 0.0  => mute this bucket
            - 0.5  => half intensity for this bucket
            - 2.0  => double intensity for this bucket (will be clipped at 15)
        """
        if 0 <= bucket_id < self.n_buckets:
            # clamp to non-negative (we don't support phase inversion)
            self.bucket_gains[bucket_id] = float(max(0.0, gain))

    # --------- RMS history for GUI ---------

    def _update_rms_history(self, rms_db: float) -> None:
        """Store last ~10s of RMS dB values for GUI stats."""
        now = time.monotonic()
        self.rms_history.append((now, float(rms_db)))
        cutoff = now - self.max_history_s
        # Drop old values
        while self.rms_history and self.rms_history[0][0] < cutoff:
            self.rms_history.pop(0)


    def get_rms_stats(self):
        """
        Return:
            (mean5, peak5, mean10, peak10) in dB.

        - mean5 / peak5  : stats over the last   5 seconds
        - mean10 / peak10: stats over the last  10 seconds

        If not enough data, values will be NaN.
        """
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

    async def send_haptic_command(self, intensity_levels: np.ndarray):
        """
        Sends commands for mapped actuators only (0..5 baseline).

        Non-additive reset:
        - duty==0 => STOP
        - duty/freq changed while running => STOP(old) then START(new)
        """
        if not self.is_connected or not self.client:
            return

        desired = self._desired_duty_per_actuator(intensity_levels)
        cmds: List[bytearray] = []

        for addr in self.mapped_actuators:
            new_duty = int(desired.get(addr, 0))

            # Choose frequency:
            # - if this actuator is associated with a bucket, use that bucket's freq
            # - otherwise fall back to the global freq_index
            bucket_id_for_act = self.actuator_to_bucket.get(addr, 0)
            if 0 <= bucket_id_for_act < len(self.freq_per_bucket):
                new_freq = self.freq_per_bucket[bucket_id_for_act]
            else:
                new_freq = self.freq_index

            new_mode = 1 if new_duty > 0 else 0

            old_mode, old_freq, old_duty = self._last_state.get(
                addr, (0, new_freq, 0)
            )

            # Decide if we must stop first (prevents stacking)
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
            packet = bytearray().join(cmds[i : i + MAX_COMMANDS_PER_PACKET])
            await self._write_packet(self._pad_packet(packet))

        self._last_ok_time = time.monotonic()

    async def stop_mapped_actuators(self):
        """
        Proper stop for ONLY mapped actuators (0..5 baseline), chunked safely.
        """
        if not self.is_connected or not self.client:
            return

        cmds: List[bytearray] = []
        for addr in self.mapped_actuators:
            old_mode, old_freq, old_duty = self._last_state.get(addr, (0, self.freq_index, 0))
            if old_mode == 1 and old_duty > 0:
                cmds.append(self.create_command(addr, mode=0, duty=0, freq=old_freq))
            self._last_state[addr] = (0, old_freq, 0)

        for i in range(0, len(cmds), MAX_COMMANDS_PER_PACKET):
            packet = bytearray().join(cmds[i : i + MAX_COMMANDS_PER_PACKET])
            await self._write_packet(self._pad_packet(packet))

    async def panic_stop(self):
        """Best-effort stop, safe to call in finally blocks and from GUI hotkey."""
        try:
            await self.stop_mapped_actuators()
        except Exception:
            pass

    # ---------------- AUDIO ----------------

    def open_microphone(self):
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_size,
        )


    def close_microphone(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            self.stream = None
            try:
                self.p.terminate()
            except Exception:
                pass

    def _frame_rms_db(self, x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean((x.astype(np.float32) ** 2)) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12)

    def process_audio_frame(self) -> Optional[np.ndarray]:
        if not self.stream:
            return None

        try:
            data = self.stream.read(self.chunk_size)
        except OSError:
            # Input overflow: drop this frame but keep running
            return None

        audio_chunk = np.frombuffer(data, dtype=np.float32)


        self.audio_buffer = np.roll(self.audio_buffer, -self.chunk_size)
        self.audio_buffer[-self.chunk_size:] = audio_chunk

        # Compute RMS in dB and update history for GUI graphs / stats
        rms_db = self._frame_rms_db(self.audio_buffer)
        self._update_rms_history(rms_db)

        # If below gate, everything off
        if rms_db < self.noise_gate_db:
            return np.zeros(self.n_buckets, dtype=np.uint8)

        bucket_energies = compute_bucket_energies(self.audio_buffer, self.cfg)

        # Choose a max duty you like (e.g. 8, 10, or 12)
        max_duty = 12

        intensity = bucket_energies_to_intensity(
            bucket_energies,
            rms_db=rms_db,
            noise_gate_db=self.noise_gate_db,
            max_duty=max_duty,
        )

         # --- APPLY PER-BUCKET GAINS (live-tunable from GUI) ---
            # Convert to float for scaling, then clip back to 0..15
        intensity = intensity.astype(np.float32)
        n = min(len(intensity), len(self.bucket_gains))
        intensity[:n] *= self.bucket_gains[:n]
        intensity = np.clip(np.round(intensity), 0, 15).astype(np.uint8)
        
        return intensity

        

    async def run_realtime_loop(self, duration_seconds: Optional[float] = None):
        """
        Realtime loop with dead-man failsafe:
        - if audio processing stalls for > idle_stop_timeout_s => stop motors
        - always panic_stop() in finally
        """
        self.open_microphone()
        self._last_ok_time = time.monotonic()
        start = time.monotonic()

        try:
            while True:
                if duration_seconds is not None and (time.monotonic() - start) > duration_seconds:
                    break

                intensity = self.process_audio_frame()

                if intensity is None:
                    if (time.monotonic() - self._last_ok_time) > self.idle_stop_timeout_s:
                        await self.panic_stop()
                        self._last_ok_time = time.monotonic()
                    await asyncio.sleep(0.01)
                    continue

                await self.send_haptic_command(intensity)
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            pass
        finally:
            await self.panic_stop()
            try:
                self.close_microphone()
            except Exception:
                pass
            try:
                if self.is_connected and self.client:
                    if sys.platform == "win32":
                        async def _disc():
                            await self.client.disconnect()
                        await self._ble_thread.run(_disc())
                    else:
                        await self.client.disconnect()
            except Exception:
                pass

            if sys.platform == "win32" and self._ble_thread:
                self._ble_thread.stop()

            self.is_connected = False
