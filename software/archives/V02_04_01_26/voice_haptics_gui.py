# voice_haptics_gui.py
"""
================================================================================
Vibraforge – GUI (Voice → Haptics) Control Panel
================================================================================

WHAT THIS GUI GIVES YOU
- Start/Stop real-time voice→haptics processing.
- ESC = PANIC STOP (immediate motor stop, without closing the app).
- Live bucket intensity display (0..15, for 6 buckets).
- Per-bucket frequency control (0..7).
- Per-bucket gain control (0.0..3.0).
- Global frequency control (apply one freq to all buckets at once).
- Global noise gate control (in dB).
- Live display of audio loudness:
    - mean + peak over the last 5 seconds
    - mean + peak over the last 10 seconds
- Presets:
    - ready-made profiles (flat, bass-heavy, treble-sparkle, mid-focus)
      that set freqs, gains and noise gate in one click.
- Help / tutorial button that explains each control.

HOW IT TALKS TO THE CONTROLLER
- GUI runs in the main Qt thread.
- HapticsWorker lives in a QThread and creates its own asyncio loop.
- HapticsWorker owns the VoiceToHapticsController instance.
- GUI sends configuration changes via methods:
    - update_bucket_freq(bucket_id, freq_index)
    - update_bucket_gain(bucket_id, gain)
    - update_noise_gate(noise_gate_db)
  which schedule updates into the worker's asyncio loop.

SAFETY
- Worker uses controller.panic_stop() in finally.
- ESC triggers panic_stop() without killing the Qt app.
- Stop button kills the loop and does a clean shutdown.

================================================================================
"""

import sys
import asyncio
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
)
from PyQt5.QtGui import QFont, QKeySequence

from voice_to_haptics_controller import VoiceToHapticsController


# Simple text bar for each bucket (0..15 -> blocks)
def bar(v: int, vmax: int = 15, width: int = 18) -> str:
    v = int(max(0, min(v, vmax)))
    filled = int(round((v / vmax) * width))
    return "█" * filled + "░" * (width - filled)


class HapticsWorker(QObject):
    # intensity array (numpy or list) -> GUI
    intensity_updated = pyqtSignal(object)
    # RMS stats: mean5, peak5, mean10, peak10 (dB)
    rms_updated = pyqtSignal(float, float, float, float)
    # status text messages
    status = pyqtSignal(str)
    # signal when worker exits
    finished = pyqtSignal()

    def __init__(self, initial_noise_gate_db: float = -60.0, initial_freq_index: int = 5):
        super().__init__()
        self._running = False
        self._loop = None
        self.controller = None

        # Initial configuration (applied when controller is created)
        self.initial_noise_gate_db = float(initial_noise_gate_db)
        self.initial_freq_index = int(initial_freq_index) & 0x07

    # ---- External control from GUI ----

    def stop(self, immediate_panic: bool = True):
        """Request the main loop to stop. Optionally send panic stop."""
        self._running = False
        if immediate_panic:
            self.panic_stop()

    def panic_stop(self):
        """
        Thread-safe panic stop:
        - schedule controller.panic_stop() inside the worker asyncio event loop.
        """
        try:
            if self._loop and self.controller and self.controller.is_connected:
                asyncio.run_coroutine_threadsafe(self.controller.panic_stop(), self._loop)
        except Exception:
            pass

    def update_bucket_freq(self, bucket_id: int, freq_index: int):
        """
        Called from GUI: change frequency index (0..7) for a given bucket.
        We schedule a small callback inside the asyncio loop.
        """
        if not (self._loop and self.controller):
            return

        def _update():
            try:
                self.controller.set_bucket_freq(bucket_id, freq_index)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_update)

    def update_noise_gate(self, noise_gate_db: float):
        """
        Called from GUI: change the global noise gate in dB.
        """
        if not (self._loop and self.controller):
            return

        def _update():
            try:
                self.controller.set_noise_gate(noise_gate_db)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_update)

    def update_bucket_gain(self, bucket_id: int, gain: float):
        """
        Called from GUI: change the gain for a given bucket.

        This schedules controller.set_bucket_gain(...) in the worker's
        asyncio loop so it's thread-safe.
        """
        if not (self._loop and self.controller):
            return

        def _update():
            try:
                self.controller.set_bucket_gain(bucket_id, gain)
            except Exception:
                pass

        self._loop.call_soon_threadsafe(_update)

    # ---- Worker main loop ----

    def run(self):
        """
        Entry point for the QThread: create an asyncio loop and run _amain().
        """
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

    async def _amain(self):
        """
        Main async routine:
        - create controller with initial config
        - connect BLE
        - open microphone
        - run real-time loop while _running is True
        """
        # Build controller with initial gate and freq
        self.controller = VoiceToHapticsController(
            sr=22050,
            n_fft=2048,
            chunk_size=1024,
            n_buckets=6,
            bucket_method="voice_landmark",
            freq_index=self.initial_freq_index,
            noise_gate_db=self.initial_noise_gate_db,
            reset_on_change=True,
        )

        self.status.emit("Scanning/connecting Vibraforge BLE…")
        ok = await self.controller.connect_vibraforge()
        if not ok:
            self.status.emit("BLE device not found. Check power/name and retry.")
            return

        self.status.emit("Connected. Opening microphone…")
        self.controller.open_microphone()
        self.status.emit("Running. Speak. ESC = PANIC STOP, Stop button = shutdown.")

        try:
            while self._running:
                intensity = self.controller.process_audio_frame()

                if intensity is None:
                    await asyncio.sleep(0.01)
                    continue

                # Send intensities to GUI
                self.intensity_updated.emit(intensity.copy())

                # Drive haptics
                await self.controller.send_haptic_command(intensity)

                # Compute and emit RMS statistics for display
                mean5, peak5, mean10, peak10 = self.controller.get_rms_stats()
                self.rms_updated.emit(mean5, peak5, mean10, peak10)

                await asyncio.sleep(0.001)

        except Exception as e:
            self.status.emit(f"Error: {e}")

        finally:
            self.status.emit("Stopping actuators…")
            try:
                await self.controller.panic_stop()
            except Exception:
                pass

            try:
                self.controller.close_microphone()
            except Exception:
                pass

            try:
                if self.controller.is_connected and self.controller.client:
                    # Your controller handles Windows MTA thread for BLE
                    await self.controller.panic_stop()  # stop motors again for safety
            except Exception:
                pass

            self.status.emit("Stopped.")


class VoiceHapticsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibraforge – Voice → Haptics Control Panel (0..5 baseline)")

        self.thread = None
        self.worker: HapticsWorker | None = None

        # store current config (used when starting worker)
        self.current_noise_gate_db: float = -60.0
        self.current_global_freq_index: int = 5

        # Presets: global freq, per-bucket freqs, per-bucket gains, noise gate.
        # You can tweak these to taste.
        self.presets = {
            "Baseline (flat)": {
                "global_freq": 5,
                "noise_gate_db": -60.0,
                "bucket_freqs": [5, 5, 5, 5, 5, 5],
                "bucket_gains": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            "Bass heavy (male voice)": {
                "global_freq": 4,
                "noise_gate_db": -55.0,
                "bucket_freqs": [3, 3, 4, 5, 5, 6],
                "bucket_gains": [2.0, 1.8, 1.2, 0.8, 0.5, 0.3],
            },
            "Mid-range speech focus": {
                "global_freq": 5,
                "noise_gate_db": -50.0,
                "bucket_freqs": [4, 4, 5, 5, 5, 6],
                "bucket_gains": [0.8, 1.0, 1.4, 1.4, 1.0, 0.6],
            },
            "Treble sparkle (lighter touch)": {
                "global_freq": 5,
                "noise_gate_db": -55.0,
                "bucket_freqs": [3, 3, 4, 5, 6, 7],
                "bucket_gains": [0.5, 0.7, 1.0, 1.3, 1.7, 2.0],
            },
        }

        # --- Layout setup ---
        w = QWidget()
        self.setCentralWidget(w)
        root = QVBoxLayout(w)

        # Status
        self.status_label = QLabel("Idle.")
        self.status_label.setFont(QFont("Arial", 11))
        root.addWidget(self.status_label)

        # Bucket intensities
        self.bucket_labels = []
        for i in range(6):
            lbl = QLabel(f"Bucket {i}: [{bar(0)}] 0/15")
            lbl.setFont(QFont("Consolas", 11))
            root.addWidget(lbl)
            self.bucket_labels.append(lbl)

        # Controls group: Start/Stop
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        root.addLayout(btn_row)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        # ESC = panic stop
        self._esc = QShortcut(QKeySequence("Esc"), self)
        self._esc.activated.connect(self.panic_stop)

        # --- Advanced controls group: Frequencies, Gains & Noise Gate ---

        ctrl_group = QGroupBox("Haptic & Audio Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # Global frequency control (0..7)
        freq_row = QHBoxLayout()
        freq_label = QLabel("Global frequency index (0–7):")
        self.global_freq_combo = QComboBox()
        for i in range(8):
            self.global_freq_combo.addItem(str(i))
        self.global_freq_combo.setCurrentIndex(self.current_global_freq_index)
        self.global_freq_combo.currentIndexChanged.connect(self.on_global_freq_changed)
        freq_row.addWidget(freq_label)
        freq_row.addWidget(self.global_freq_combo)
        ctrl_layout.addLayout(freq_row)

        # Per-bucket frequency + gain controls
        self.freq_combos = []
        self.gain_spins = []

        for i in range(6):
            row = QHBoxLayout()

            # Label
            lbl = QLabel(f"Bucket {i}:")
            row.addWidget(lbl)

            # Frequency combobox (0..7)
            combo = QComboBox()
            for k in range(8):
                combo.addItem(str(k))
            combo.setCurrentIndex(self.current_global_freq_index)
            combo.currentIndexChanged.connect(
                lambda idx, b=i: self.on_bucket_freq_changed(b, idx)
            )
            row.addWidget(QLabel("freq:"))
            row.addWidget(combo)

            # Gain spinbox (0.0 .. 3.0)
            gain_spin = QDoubleSpinBox()
            gain_spin.setRange(0.0, 3.0)
            gain_spin.setSingleStep(0.1)
            gain_spin.setValue(1.0)  # neutral
            gain_spin.valueChanged.connect(
                lambda val, b=i: self.on_bucket_gain_changed(b, val)
            )
            row.addWidget(QLabel("gain:"))
            row.addWidget(gain_spin)

            ctrl_layout.addLayout(row)
            self.freq_combos.append(combo)
            self.gain_spins.append(gain_spin)

        # Noise gate control (dB)
        gate_row = QHBoxLayout()
        self.noise_label = QLabel(f"Noise gate (dB): {self.current_noise_gate_db:.1f}")
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(-80.0, -20.0)
        self.noise_spin.setSingleStep(1.0)
        self.noise_spin.setValue(self.current_noise_gate_db)
        self.noise_spin.valueChanged.connect(self.on_noise_gate_changed)
        gate_row.addWidget(self.noise_label)
        gate_row.addWidget(self.noise_spin)
        ctrl_layout.addLayout(gate_row)

        root.addWidget(ctrl_group)

        # --- Presets + Help group ---

        preset_group = QGroupBox("Presets & Help")
        preset_layout = QHBoxLayout(preset_group)

        preset_label = QLabel("Preset:")
        self.preset_combo = QComboBox()
        for name in self.presets.keys():
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)

        self.help_btn = QPushButton("Help / What does this do?")
        self.help_btn.clicked.connect(self.show_help)

        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch(1)
        preset_layout.addWidget(self.help_btn)

        root.addWidget(preset_group)

        # --- RMS stats display ---

        rms_group = QGroupBox("Audio loudness (RMS dB)")
        rms_layout = QVBoxLayout(rms_group)

        self.rms_label_5s = QLabel("5s mean: --  peak: --")
        self.rms_label_10s = QLabel("10s mean: --  peak: --")

        self.rms_label_5s.setFont(QFont("Consolas", 10))
        self.rms_label_10s.setFont(QFont("Consolas", 10))

        rms_layout.addWidget(self.rms_label_5s)
        rms_layout.addWidget(self.rms_label_10s)

        root.addWidget(rms_group)

    # ------------- GUI Event Handlers -------------

    def start(self):
        """Start worker in a background QThread."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = HapticsWorker(
            initial_noise_gate_db=self.current_noise_gate_db,
            initial_freq_index=self.current_global_freq_index,
        )
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

    def stop(self):
        """Stop processing and shutdown worker."""
        if self.worker:
            self.worker.stop(immediate_panic=True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping…")

    def panic_stop(self):
        """Emergency motor stop (ESC)."""
        if self.worker:
            self.worker.panic_stop()
        self.status_label.setText("PANIC STOP sent (ESC).")

    def _on_stopped(self):
        self.worker = None
        self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Idle (stopped).")

    def on_intensity(self, arr):
        """Update bucket intensity labels."""
        try:
            arr = np.asarray(arr).astype(int)
            for i in range(min(6, len(arr))):
                v = int(arr[i])
                self.bucket_labels[i].setText(f"Bucket {i}: [{bar(v)}] {v}/15")
        except Exception as e:
            self.status_label.setText(f"UI update error: {e}")

    def on_rms_updated(self, mean5, peak5, mean10, peak10):
        """Update RMS statistics labels."""
        def fmt(x):
            if np.isnan(x):
                return "--"
            return f"{x:5.1f} dB"

        self.rms_label_5s.setText(
            f"5s mean: {fmt(mean5)}   peak: {fmt(peak5)}"
        )
        self.rms_label_10s.setText(
            f"10s mean: {fmt(mean10)}  peak: {fmt(peak10)}"
        )

    def on_global_freq_changed(self, idx: int):
        """
        Global frequency combobox changed:
        - update internal current_global_freq_index
        - update all per-bucket combos
        - tell worker to update all bucket freqs (if running)
        """
        self.current_global_freq_index = int(idx)

        # Update each bucket combo (this will also trigger on_bucket_freq_changed)
        for i, combo in enumerate(self.freq_combos):
            if combo.currentIndex() != idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
                # manually propagate now
                self.on_bucket_freq_changed(i, idx)

    def on_bucket_freq_changed(self, bucket_id: int, idx: int):
        """
        Per-bucket frequency changed.
        We tell the worker to update controller.set_bucket_freq().
        """
        if self.worker:
            self.worker.update_bucket_freq(bucket_id, int(idx))

    def on_bucket_gain_changed(self, bucket_id: int, value: float):
        """
        Called whenever a bucket's gain spinbox is changed in the GUI.
        We forward the new gain to the worker, which updates the controller.
        """
        if self.worker:
            self.worker.update_bucket_gain(bucket_id, float(value))

    def on_noise_gate_changed(self, value: float):
        """Noise gate spinbox changed."""
        self.current_noise_gate_db = float(value)
        self.noise_label.setText(f"Noise gate (dB): {self.current_noise_gate_db:.1f}")
        if self.worker:
            self.worker.update_noise_gate(self.current_noise_gate_db)

    def on_preset_changed(self, idx: int):
        """
        Apply a preset:
        - global freq
        - per-bucket freqs
        - per-bucket gains
        - noise gate
        """
        name = self.preset_combo.itemText(idx)
        cfg = self.presets.get(name)
        if not cfg:
            return

        # 1) Global frequency
        gf = int(cfg.get("global_freq", self.current_global_freq_index))
        self.current_global_freq_index = gf
        self.global_freq_combo.blockSignals(True)
        self.global_freq_combo.setCurrentIndex(gf)
        self.global_freq_combo.blockSignals(False)

        # 2) Per-bucket frequencies
        freqs = cfg.get("bucket_freqs", [])
        for i, combo in enumerate(self.freq_combos):
            if i < len(freqs):
                val = int(freqs[i])
            else:
                val = gf
            combo.blockSignals(True)
            combo.setCurrentIndex(val)
            combo.blockSignals(False)
            self.on_bucket_freq_changed(i, val)

        # 3) Per-bucket gains
        gains = cfg.get("bucket_gains", [])
        for i, spin in enumerate(self.gain_spins):
            if i < len(gains):
                g = float(gains[i])
            else:
                g = 1.0
            spin.blockSignals(True)
            spin.setValue(g)
            spin.blockSignals(False)
            self.on_bucket_gain_changed(i, g)

        # 4) Noise gate
        gate = float(cfg.get("noise_gate_db", self.current_noise_gate_db))
        self.current_noise_gate_db = gate
        self.noise_spin.blockSignals(True)
        self.noise_spin.setValue(gate)
        self.noise_spin.blockSignals(False)
        self.noise_label.setText(f"Noise gate (dB): {gate:.1f}")
        if self.worker:
            self.worker.update_noise_gate(gate)

        self.status_label.setText(f"Preset applied: {name}")

    def show_help(self):
        """
        Show a simple textual tutorial about what each control does.
        """
        text = (
            "Vibraforge Voice→Haptics GUI – quick tour:\n\n"
            "Start / Stop:\n"
            "  • Start: connects to the Vibraforge controller and starts listening to the microphone.\n"
            "  • Stop: stops processing and sends a stop command to the actuators.\n"
            "  • ESC key: PANIC STOP – immediately stops all actuators but keeps the app open.\n\n"
            "Bucket bars:\n"
            "  • Each line 'Bucket 0..5' shows the current intensity (0–15) for one frequency band of your voice.\n"
            "  • Buckets are mapped to actuators 0..5 on your Vibraforge prototype.\n\n"
            "Global frequency index:\n"
            "  • Sets a base vibration frequency index (0–7).\n"
            "  • Changing it updates all bucket frequencies at once.\n"
            "  • Lower values usually feel slower/deeper, higher values faster/higher.\n\n"
            "Per-bucket 'freq':\n"
            "  • Frequency index for this specific bucket/actuator.\n"
            "  • Overrides the global frequency for that bucket.\n\n"
            "Per-bucket 'gain':\n"
            "  • Multiplicative factor on that bucket's intensity.\n"
            "  • 1.0 = normal, 0.5 = weaker, 0.0 = muted, 2.0 = stronger.\n\n"
            "Noise gate (dB):\n"
            "  • Below this loudness level, everything is treated as silence (no vibration).\n"
            "  • Raise the value (e.g. to -50 dB) if you only want strong voice to trigger haptics.\n"
            "  • Lower the value (e.g. to -65 dB) if you want very soft sounds to be felt.\n\n"
            "RMS loudness (5s / 10s):\n"
            "  • Shows the mean and peak loudness of your voice over the last 5 and 10 seconds.\n"
            "  • Use it to see how loud you are speaking and how it relates to the noise gate.\n\n"
            "Presets:\n"
            "  • 'Baseline (flat)': all buckets same freq and gain, low noise gate.\n"
            "  • 'Bass heavy (male voice)': stronger low buckets, weaker highs.\n"
            "  • 'Mid-range speech focus': emphasizes typical speech bands.\n"
            "  • 'Treble sparkle': lighter low-end with more high-frequency accents.\n"
            "  • You can still tweak any control after applying a preset.\n"
        )
        QMessageBox.information(self, "Vibraforge – Quick tutorial", text)

    def closeEvent(self, event):
        """Ensure motors are stopped when closing the window."""
        self.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    gui = VoiceHapticsGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
