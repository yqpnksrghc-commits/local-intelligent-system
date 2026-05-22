"""
Microphone capture → transcribed text via faster-whisper (fully local).
Silence-gated: records until the speaker stops, then returns the transcript.
"""
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def _get_model(size: str = "base") -> WhisperModel:
    global _model
    if _model is None:
        print(f"[listener] loading Whisper {size} …")
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def record(
    samplerate: int = 16000,
    silence_threshold: float = 0.01,
    silence_duration: float = 1.2,
    max_duration: float = 30.0,
) -> np.ndarray:
    """
    Record from the default microphone.
    Stops when RMS stays below silence_threshold for silence_duration seconds,
    or after max_duration seconds.
    """
    chunk = int(samplerate * 0.1)   # 100 ms frames
    silence_frames = int(silence_duration / 0.1)
    max_frames = int(max_duration / 0.1)

    frames: list[np.ndarray] = []
    silent_count = 0
    started = False

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        print("[listener] listening …", flush=True)
        for _ in range(max_frames):
            data, _ = stream.read(chunk)
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > silence_threshold:
                started = True
                silent_count = 0
                frames.append(data.copy())
            elif started:
                frames.append(data.copy())
                silent_count += 1
                if silent_count >= silence_frames:
                    break

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def transcribe(audio: np.ndarray, samplerate: int = 16000, model_size: str = "base") -> tuple[str, str]:
    """
    Returns (transcript, detected_language).
    """
    model = _get_model(model_size)
    segments, info = model.transcribe(audio, beam_size=5)
    text = " ".join(s.text.strip() for s in segments)
    return text.strip(), info.language
