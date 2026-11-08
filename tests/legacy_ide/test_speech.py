"""Unit tests for local Hindi and Marathi speech synthesis."""

import pytest
import soundfile as sf
try:
    import torch
except ImportError:
    torch = None

from modules.speech import SPEECH_MODELS, SpeechSynthesizer


def test_empty_input_is_rejected_without_loading_model(tmp_path, monkeypatch):
    synthesizer = SpeechSynthesizer(output_dir=tmp_path)
    monkeypatch.setattr(
        synthesizer,
        "load_model",
        lambda language: (_ for _ in ()).throw(AssertionError("model should not load")),
    )

    result = synthesizer.synthesize("  \n ", "Hindi")

    assert result["success"] is False
    assert result["audio_path"] is None
    assert "empty" in result["error"].lower()


def test_unsupported_language_is_rejected(tmp_path):
    result = SpeechSynthesizer(output_dir=tmp_path).synthesize("नमस्ते", "French")

    assert result["success"] is False
    assert "unsupported language" in result["error"].lower()


def test_synthesis_writes_readable_wav_and_returns_metadata(tmp_path, monkeypatch):
    synthesizer = SpeechSynthesizer(output_dir=tmp_path, default_device="cpu")

    class FakeModel:
        def generate_speech(self, text, tokenizer):
            assert text == "नमस्ते"
            assert tokenizer == "fake-tokenizer"
            return torch.linspace(-0.2, 0.2, 2400)

    monkeypatch.setattr(
        synthesizer,
        "load_model",
        lambda language: (FakeModel(), "fake-tokenizer"),
    )

    result = synthesizer.synthesize(" नमस्ते ", "hi")

    assert result["success"] is True
    assert result["language"] == "Hindi"
    assert result["model_name"] == SPEECH_MODELS["Hindi"]["model_id"]
    assert result["device"] == "cpu"
    assert result["latency_seconds"] >= 0
    assert result["duration_seconds"] == 0.1
    audio, sample_rate = sf.read(result["audio_path"])
    assert sample_rate == 24_000
    assert len(audio) == 2400


def test_model_loading_passes_local_token_and_trusts_official_checkpoint(
    tmp_path, monkeypatch
):
    synthesizer = SpeechSynthesizer(output_dir=tmp_path, default_device="cpu")
    calls = {}

    class FakeTokenizer:
        pass

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    monkeypatch.setattr("modules.speech.get_huggingface_token", lambda: "hf_test_token")
    monkeypatch.setattr(
        "modules.speech.AutoTokenizer.from_pretrained",
        lambda model_id, **kwargs: calls.update(tokenizer=(model_id, kwargs)) or FakeTokenizer(),
    )
    monkeypatch.setattr(
        "modules.speech.AutoModelForCausalLM.from_pretrained",
        lambda model_id, **kwargs: calls.update(model=(model_id, kwargs)) or FakeModel(),
    )

    model, tokenizer = synthesizer.load_model("Marathi")

    assert isinstance(tokenizer, FakeTokenizer)
    assert isinstance(model, FakeModel)
    assert calls["tokenizer"][0] == SPEECH_MODELS["Marathi"]["model_id"]
    assert calls["tokenizer"][1]["token"] == "hf_test_token"
    assert calls["tokenizer"][1]["trust_remote_code"] is True
    assert calls["model"][1]["token"] == "hf_test_token"
    assert calls["model"][1]["trust_remote_code"] is True
    assert synthesizer.is_model_loaded()


def test_memory_error_returns_actionable_failure(tmp_path, monkeypatch):
    synthesizer = SpeechSynthesizer(output_dir=tmp_path)

    class FakeModel:
        def generate_speech(self, text, tokenizer):
            raise MemoryError("test memory error")

    monkeypatch.setattr(
        synthesizer,
        "load_model",
        lambda language: (FakeModel(), "fake-tokenizer"),
    )

    result = synthesizer.synthesize("नमस्कार", "Marathi")

    assert result["success"] is False
    assert result["audio_path"] is None
    assert "insufficient memory" in result["error"].lower()


def test_dictionary_audio_tensor_is_squeezed_into_mono_wav(tmp_path, monkeypatch):
    if torch is None:
        pytest.skip("torch is not installed in the environment")
    synthesizer = SpeechSynthesizer(output_dir=tmp_path, default_device="cpu")

    class FakeModel:
        def generate_speech(self, text, tokenizer):
            return {"audio": torch.linspace(-0.2, 0.2, 2400).unsqueeze(0)}

    monkeypatch.setattr(
        synthesizer,
        "load_model",
        lambda language: (FakeModel(), "fake-tokenizer"),
    )

    result = synthesizer.synthesize("नमस्कार", "Hindi")

    assert result["success"] is True
    audio, sample_rate = sf.read(result["audio_path"])
    assert sample_rate == 24_000
    assert audio.ndim == 1
    assert len(audio) == 2400


def test_synthesis_error_does_not_log_or_return_local_token(tmp_path, monkeypatch, caplog):
    synthesizer = SpeechSynthesizer(output_dir=tmp_path, default_device="cpu")
    secret = "hf_test_secret"

    class FakeModel:
        def generate_speech(self, text, tokenizer):
            raise RuntimeError(f"request failed with {secret}")

    monkeypatch.setattr("modules.speech.get_huggingface_token", lambda: secret)
    monkeypatch.setattr(
        synthesizer,
        "load_model",
        lambda language: (FakeModel(), "fake-tokenizer"),
    )

    result = synthesizer.synthesize("नमस्कार", "Hindi")

    assert result["success"] is False
    assert secret not in result["error"]
    assert secret not in caplog.text
