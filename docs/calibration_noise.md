Important: disable noise reduction / “audio enhancements”
TimbreForge relies on spectral energy. Any denoiser / noise suppression / AGC can distort the spectrum and break colour activation.
Before running:

Disable Windows microphone Enhancements (and any “Noise suppression”, “AGC”, “Echo cancellation”).

Avoid running through apps that add suppression (Zoom/Teams/Discord/NVIDIA Broadcast) unless you explicitly want that behaviour.
TimbreForge uses ambient-noise calibration + per-colour noise-floor subtraction (not “studio denoising”).


For voice capture, the usual strategy is to record a clean signal (treated room + good mic technique) rather than applying aggressive noise reduction during recording.
TimbreForge follows this philosophy: keep the signal path clean, then apply light calibration for real-room robustness.


# Calibration & noise (why we do it)

TimbreForge uses a short **ambient-noise calibration** and **per-colour noise-floor subtraction**
to reduce ghost vibrations in real rooms.

!!! warning "Disable denoisers first"
    Noise suppressors / AGC can distort spectral energy and break colour activation.
    Prefer a clean signal path + calibration.

## Studio note
Typically, professional voice capture aims for a clean recording (room + mic technique) rather than heavy noise reduction during recording.
