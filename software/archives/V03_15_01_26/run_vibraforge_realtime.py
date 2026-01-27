# run_vibraforge_realtime.py
"""
================================================================================
Headless baseline runner (no GUI)

Use this when you want the simplest end-to-end test:
- Connect BLE
- Open mic
- Run until Ctrl+C
- Always stop motors on exit (panic_stop in finally)

If you need to change which motors move:
- Edit default_actuator_mapping_m6() in voice_to_haptics_controller.py
================================================================================
"""
import sys
sys.coinit_flags = 0  # 0 = MTA (needed for Bleak WinRT console apps on Windows)

try:
    from bleak.backends.winrt.util import uninitialize_sta
    uninitialize_sta()  # undo STA if some library already set it
except Exception:
    pass


import asyncio
from voice_to_haptics_controller import VoiceToHapticsController

async def main():
    controller = VoiceToHapticsController(
        sr=22050,
        n_fft=2048,
        chunk_size=1024,
        n_buckets=6,
        bucket_method="voice_landmark",
        freq_index=5,
        noise_gate_db=-60.0,
    )

    ok = await controller.connect_vibraforge()
    if not ok:
        print("Could not connect to Vibraforge (check name & power)")
        return

    await controller.run_realtime_loop(duration_seconds=None)

if __name__ == "__main__":
    asyncio.run(main())
