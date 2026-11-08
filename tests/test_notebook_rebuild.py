"""Notebook-derived reconstruction unit tests (no model downloads)."""
from __future__ import annotations

import pandas as pd

from modules.bias_evaluation import (
    compare_neutral_professions,
    create_gender_pair_comparison,
    detect_gender_markers,
    run_bias_evaluation,
    summarize_bias_results,
)
from modules.evaluation import calculate_metrics, evaluate_against_references, run_model_comparison
from modules.image_translation import process_image
from modules.prompt_engineering import (
    SUPPORTED_TECHNIQUES,
    build_translation_prompt,
    compare_prompt_techniques,
    validate_inputs,
)


def test_prompt_techniques_and_styles():
    for tech in SUPPORTED_TECHNIQUES:
        p = build_translation_prompt("Hello world", "Hindi", "Formal", tech)
        assert "Hindi" in p and "Hello world" in p
    prompts = compare_prompt_techniques("Test", "Marathi", "Academic")
    assert set(prompts) == set(SUPPORTED_TECHNIQUES)


def test_prompt_validation():
    try:
        validate_inputs("", "Hindi")
        assert False, "should raise"
    except ValueError:
        pass


def test_gender_markers_hindi_marathi():
    hi = detect_gender_markers("वह काम करता है।", "Hindi")
    assert hi["masculine_count"] > 0
    mr = detect_gender_markers("ती काम करते.", "Marathi")
    assert mr["feminine_count"] > 0


def test_bias_evaluation_pipeline():
    df = pd.DataFrame([
        {"id": "1", "gender_condition": "neutral", "profession": "doctor", "english": "The doctor works."},
        {"id": "2", "gender_condition": "feminine", "profession": "doctor", "english": "She is a doctor."},
    ])

    def mock(text, lang):
        return {"success": True, "translated": "डॉक्टर काम करता है।" if lang == "Hindi" else "डॉक्टर काम करतो."}

    res = run_bias_evaluation(df, mock, "Hindi", "mock")
    assert len(res) == 2 and res["success"].all()
    assert not summarize_bias_results(res).empty
    assert compare_neutral_professions(res) is not None


def test_evaluation_missing_references():
    results = pd.DataFrame([{"id": 1, "model": "m", "success": True, "translation": "नमस्ते"}])
    bench = pd.DataFrame([{"id": 1, "reference_hindi": ""}])
    out = evaluate_against_references(results, "reference_hindi", bench)
    status = str(out["status"].iloc[0])
    assert "Reference translations missing" in status


def test_evaluation_with_references():
    m = calculate_metrics(["नमस्ते दुनिया"], ["नमस्ते दुनिया"])
    assert "BLEU" in m and "chrF" in m


def test_model_comparison_mock():
    def ok(text, lang):
        return {"success": True, "translated": "x", "latency_seconds": 0.01, "device": "cpu"}

    df = run_model_comparison(["a", "b"], {"M": ok}, "Hindi")
    assert len(df) == 2 and df["success"].all()
    assert "error" in df.columns


def test_image_pipeline_mock(monkeypatch):
    import modules.image_translation as it

    monkeypatch.setattr(it, "caption_image", lambda p: {"success": True, "caption": "a cat", "model": "blip", "latency_seconds": 0.1})
    res = process_image("x.jpg", "Hindi", lambda text, lang: {"success": True, "translated": "बिल्ली", "model": "m"})
    assert res["success"] and res["english_caption"] == "a cat"
