# voice_haptics_gui.py
"""
================================================================================
Vibraforge – GUI (Voice → Haptics) + PANIC STOP HOTKEY
================================================================================

WHAT THIS GUI IS FOR
- Minimal UI for testing:
  - Start/Stop
  - Live bars showing bucket intensities (0..15)
  - BLE + Mic are run in a worker thread (QThread) so the UI stays responsive

KEY SAFETY FEATURE
- Press ESC at any time = PANIC STOP.
  This schedules controller.panic_stop() into the worker's asyncio loop.

WHAT TO MODIFY
- Change the frequency index in VoiceToHapticsController(...) if desired
- Change the controller mapping in voice_to_haptics_controller.py (baseline 0..5)
- If you see buzzing at silence, raise noise_gate_db in the controller

================================================================================
"""
import sys
sys.coinit_flags = 0  # 0 = MTA, required for Bleak WinRT in console apps on Windows


import sys
import asyncio
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QShortcut
)
from PyQt5.QtGui import QFont, QKeySequence

from voice_to_haptics_controller import VoiceToHapticsController


def bar(v: int, vmax: int = 15, width: int = 18) -> str:
    v = int(max(0, min(v, vmax)))
    filled = int(round((v / vmax) * width))
    return "█" * filled + "░" * (width - filled)


class HapticsWorker(QObject):
    intensity_updated = pyqtSignal(object)
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = False
        self._loop = None
        self.controller = None

    def stop(self, immediate_panic: bool = True):
        self._running = False
        if immediate_panic:
            self.panic_stop()

    def panic_stop(self):
        """
        Thread-safe panic stop:
        schedule controller.panic_stop() inside the worker asyncio event loop.
        """
        try:
            if self._loop and self.controller and self.controller.is_connected:
                asyncio.run_coroutine_threadsafe(self.controller.panic_stop(), self._loop)
        except Exception:
            pass

    def run(self):
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
        # You can tweak freq_index/noise_gate_db here for quick experiments:
        self.controller = VoiceToHapticsController(
            sr=22050,
            n_fft=2048,
            chunk_size=1024,
            n_buckets=6,
            freq_index=5,
            noise_gate_db=-60.0,
            reset_on_change=True,
        )

        self.status.emit("Scanning/connecting BLE…")
        ok = await self.controller.connect_vibraforge()
        if not ok:
            self.status.emit("BLE device not found (name/advertising).")
            return

        self.status.emit("Connected. Opening microphone…")
        self.controller.open_microphone()
        self.status.emit("Running. Speak. ESC = PANIC STOP. Stop button = stop & shutdown.")

        try:
            while self._running:
                intensity = self.controller.process_audio_frame()
                if intensity is None:
                    await asyncio.sleep(0.01)
                    continue

                self.intensity_updated.emit(intensity.copy())
                await self.controller.send_haptic_command(intensity)

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
                    await self.controller.client.disconnect()
            except Exception:
                pass

            self.status.emit("Stopped.")


class VoiceHapticsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibraforge – Voice → Haptics (Baseline 0..5)")

        self.thread = None
        self.worker = None

        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)

        self.status_label = QLabel("Idle.")
        self.status_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.status_label)

        self.bucket_labels = []
        for i in range(6):
            lbl = QLabel(f"Bucket {i}: [{bar(0)}] 0/15")
            lbl.setFont(QFont("Consolas", 11))
            layout.addWidget(lbl)
            self.bucket_labels.append(lbl)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        # ESC = panic stop (even if UI is focused somewhere)
        self._esc = QShortcut(QKeySequence("Esc"), self)
        self._esc.activated.connect(self.panic_stop)

    def start(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = HapticsWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.intensity_updated.connect(self.on_intensity)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_stopped)

        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop(immediate_panic=True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping…")

    def panic_stop(self):
        # Does NOT close the program; it just stops motors immediately.
        if self.worker:
            self.worker.panic_stop()
        self.status_label.setText("PANIC STOP sent (ESC).")

    def _on_stopped(self):
        self.worker = None
        self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_intensity(self, arr):
        try:
            arr = np.asarray(arr).astype(int)
            for i in range(min(6, len(arr))):
                v = int(arr[i])
                self.bucket_labels[i].setText(f"Bucket {i}: [{bar(v)}] {v}/15")
        except Exception as e:
            self.status_label.setText(f"UI update error: {e}")

    def closeEvent(self, event):
        # Always try to stop motors when closing the window
        self.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    gui = VoiceHapticsGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
