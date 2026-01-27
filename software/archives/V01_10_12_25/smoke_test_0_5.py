import asyncio
import numpy as np
import sys
from voice_to_haptics_controller import VoiceToHapticsController

async def main():
    c = VoiceToHapticsController(n_buckets=6, freq_index=5)
    ok = await c.connect_vibraforge()
    if not ok:
        print("Not connected")
        return

    try:
        for addr in range(6):
            print("Testing addr", addr)
            intensity = np.zeros(6, dtype=np.uint8)
            intensity[addr] = 12
            await c.send_haptic_command(intensity)
            await asyncio.sleep(1.0)
            await c.panic_stop()
            await asyncio.sleep(0.3)
    finally:
        await c.panic_stop()
        try:
            if c.is_connected and c.client:
                if sys.platform == "win32":
                    async def _disc():
                        await c.client.disconnect()
                    await c._ble_thread.run(_disc())
                else:
                    await c.client.disconnect()
        except Exception:
            pass
        if sys.platform == "win32" and c._ble_thread:
            c._ble_thread.stop()

asyncio.run(main())
