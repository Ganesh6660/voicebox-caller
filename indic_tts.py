"""
Runtime inference for the quantized Hindi/Kannada MMS-TTS models produced by
quantize_export.py (run that script once before importing this in server.py
with language="hi" or "kn").

Models are loaded lazily and cached in memory, so the ~15-40MB int8 ONNX
file for a given language is only read from disk once per server process.
"""

import io
import json
import os
import wave

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_session_cache = {}
_tokenizer_cache = {}
_meta_cache = {}


def _load(language: str):
    if language not in _session_cache:
        model_dir = os.path.join(MODELS_DIR, language)
        onnx_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"No quantized model found for language '{language}' at {onnx_path}. "
                "Run `python quantize_export.py` once before using this language."
            )
        _session_cache[language] = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        _tokenizer_cache[language] = AutoTokenizer.from_pretrained(model_dir)
        with open(os.path.join(model_dir, "meta.json")) as f:
            _meta_cache[language] = json.load(f)

    return _session_cache[language], _tokenizer_cache[language], _meta_cache[language]


def synthesize_indic(text: str, language: str):
    """Returns (waveform: float32 numpy array in [-1, 1], sample_rate: int)."""
    session, tokenizer, meta = _load(language)
    inputs = tokenizer(text, return_tensors="np")

    outputs = session.run(
        ["waveform"],
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )
    waveform = outputs[0].squeeze()
    return waveform, meta["sampling_rate"]


def synthesize_indic_wav_bytes(text: str, language: str) -> bytes:
    """Convenience wrapper returning ready-to-use WAV bytes — same shape as
    Voicebox's /generate response, so server.py's downstream resampling code
    (which expects WAV bytes) doesn't need to care which engine produced them."""
    waveform, sample_rate = synthesize_indic(text, language)
    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()
