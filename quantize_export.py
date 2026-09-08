"""
One-time setup: export Meta's MMS-TTS Hindi and Kannada models to ONNX and
quantize them to int8 for fast CPU inference.

Run this once, locally, before using --language hi/kn with place_call.py:

    pip install torch transformers onnx onnxruntime
    python quantize_export.py

It downloads facebook/mms-tts-hin and facebook/mms-tts-kan from Hugging
Face (a few hundred MB total, one-time), exports each to ONNX, then
dynamically quantizes to int8. Output lands in ./models/<lang>/ — that's
what indic_tts.py reads at runtime. You only need to run this once; after
that, server.py just loads the quantized .onnx files directly and never
needs torch again (torch/onnx are export-time only dependencies).

NOTE ON LICENSE: facebook/mms-tts-hin and facebook/mms-tts-kan are released
under CC-BY-NC 4.0 — non-commercial use only. That's fine for personal
testing, but if this project ever goes commercial, swap in
AI4Bharat/Indic-TTS (MIT licensed) instead — see README.md.
"""

import json
import os

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer, VitsModel

MODELS = {
    "hi": "facebook/mms-tts-hin",
    "kn": "facebook/mms-tts-kan",
}

# Each MMS-TTS tokenizer's vocab is script-specific (Devanagari for Hindi, Kannada
# script for Kannada) — an English dummy sentence tokenizes to an empty tensor
# (no Latin characters are in the vocab), which breaks ONNX tracing with a
# confusing "Expected tensor ... Long, Int; but got FloatTensor" error deep in
# the embedding layer. Use real text in each language's own script instead.
DUMMY_TEXT = {
    "hi": "नमस्ते, आप कैसे हैं?",
    "kn": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")


class VitsONNXWrapper(torch.nn.Module):
    """Wraps VitsModel so torch.onnx.export sees a plain tensor-in/tensor-out forward
    (VitsModel.forward normally returns a dataclass, not a bare tensor)."""

    def __init__(self, model: VitsModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).waveform


def export_and_quantize(lang_code: str, hf_model_id: str):
    out_dir = os.path.join(OUTPUT_DIR, lang_code)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{lang_code}] downloading {hf_model_id} ...")
    model = VitsModel.from_pretrained(hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model.eval()

    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"sampling_rate": model.config.sampling_rate}, f)

    wrapper = VitsONNXWrapper(model)
    dummy_inputs = tokenizer(DUMMY_TEXT[lang_code], return_tensors="pt")

    fp32_path = os.path.join(out_dir, "model.onnx")
    int8_path = os.path.join(out_dir, "model.int8.onnx")

    print(f"[{lang_code}] exporting to ONNX ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            fp32_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["waveform"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "waveform": {0: "batch", 1: "time"},
            },
            opset_version=17,
        )
    onnx.checker.check_model(fp32_path)

    print(f"[{lang_code}] quantizing to int8 ...")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)

    fp32_size = os.path.getsize(fp32_path) / 1e6
    int8_size = os.path.getsize(int8_path) / 1e6
    print(f"[{lang_code}] done: {fp32_size:.1f}MB -> {int8_size:.1f}MB  ({out_dir})")

    # Only the quantized version is needed at runtime.
    os.remove(fp32_path)


if __name__ == "__main__":
    for lang, hf_id in MODELS.items():
        export_and_quantize(lang, hf_id)
    print("\nDone. server.py will pick these up automatically from ./models/<lang>/")
