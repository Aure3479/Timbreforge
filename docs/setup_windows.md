# Setup (Windows)

## Recommended install (conda)
```bash
conda create -n timbreforge python=3.10 -y
conda activate timbreforge
conda install -c conda-forge numpy pyaudio -y
pip install -r requirements.txt
```

## Run
```bash
python software/current/voice_haptics_gui_patched_phase1_debugtab_v4.py
```

## Important (disable audio processing)

- Disable Windows mic Enhancements (noise suppression / AGC / echo).
- Avoid running through Zoom/Teams suppression unless you test that specifically.