"""
Unit and Functional Tests for Phase 4: Model Comparison & Human Rating Studio.
Validates multi-pipeline orchestration, human rating calculations, DataFrame formatting, and CSV exports.
"""

from unittest.mock import patch
import pandas as pd
import pytest

from modules.evaluation import (
    DISCLAIMER_TEXT,
    EVALUATION_CRITERIA,
    calculate_bleu_chrf,
    calculate_rating_average,
    compute_sentence_metrics,
    create_comparison_dataframe,
    export_benchmark_outputs,
    export_comparison_and_ratings_to_csv,
    load_benchmark_dataset,
    run_benchmark_evaluation,
    run_model_comparison,
)


# ============================================================================
# 1. Human Rating Calculation Tests
# ============================================================================

def test_calculate_rating_average_all_criteria():
    """Verify exact arithmetic mean calculation across all 5 evaluation criteria."""
    ratings = {
        "meaning_preservation": 5.0,
        "grammar": 4.0,
        "fluency": 4.5,
        "naturalness": 3.5,
        "overall_usefulness": 4.0,
    }
    # Expected mean: (5.0 + 4.0 + 4.5 + 3.5 + 4.0) / 5 = 21.0 / 5 = 4.2
    avg = calculate_rating_average(ratings)
    assert avg == 4.2


def test_calculate_rating_average_empty_returns_none():
    """Verify that an empty ratings dictionary returns None (no fabricated default scores)."""
    assert calculate_rating_average({}) is None


def test_calculate_rating_average_partial_criteria():
    """Verify average calculation when only a subset of criteria are provided."""
    ratings = {
        "meaning_preservation": 4.0,
        "grammar": 5.0,
    }
    avg = calculate_rating_average(ratings)
    assert avg == 4.5


# ============================================================================
# 2. Comparison Dataframe and CSV Export Tests
# ============================================================================

@pytest.fixture
def sample_comparison_payload():
    """Fixture providing a mock comparison result for testing export formats."""
    return {
        "original_text": "Good morning.",
        "target_language": "Hindi",
        "selected_tone": "formal",
        "timestamp": 1700000000.0,
        "candidates": [
            {
                "key": "marian",
                "display_name": "MarianMT (Helsinki-NLP)",
                "architecture": "Seq2Seq (6-Enc / 6-Dec)",
                "technique": "Deterministic Beam Search",
                "tone": "Neutral",
                "translation": "सुप्रभात।",
                "latency_seconds": 0.045,
                "device": "cuda",
                "success": True,
            },
            {
                "key": "qwen_zero_shot",
                "display_name": "Qwen2.5-1.5B (Zero-Shot)",
                "architecture": "Decoder-Only Causal LM",
                "technique": "Zero-Shot Direct Translation",
                "tone": "Neutral",
                "translation": "सुप्रभात।",
                "latency_seconds": 0.320,
                "device": "cuda",
                "success": True,
            },
            {
                "key": "qwen_few_shot",
                "display_name": "Qwen2.5-1.5B (Few-Shot)",
                "architecture": "Decoder-Only Causal LM",
                "technique": "Few-Shot In-Context Learning",
                "tone": "Neutral",
                "translation": "शुभ प्रभात।",
                "latency_seconds": 0.410,
                "device": "cuda",
                "success": True,
            },
            {
                "key": "qwen_tone_controlled",
                "display_name": "Qwen2.5-1.5B (Tone: Formal)",
                "architecture": "Decoder-Only Causal LM",
                "technique": "Role-Based Register Directives (Formal)",
                "tone": "Formal",
                "translation": "सुप्रभात, आपका दिन शुभ हो।",
                "latency_seconds": 0.390,
                "device": "cuda",
                "success": True,
            },
        ],
    }


def test_create_comparison_dataframe(sample_comparison_payload):
    """Verify that create_comparison_dataframe outputs a valid, populated DataFrame."""
    df = create_comparison_dataframe(sample_comparison_payload)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    expected_cols = [
        "Model Name",
        "Architecture",
        "Prompt Technique",
        "Target Language",
        "Translation",
        "Latency (s)",
        "Device",
        "Success",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df["Model Name"].iloc[0] == "MarianMT (Helsinki-NLP)"
    assert df["Target Language"].iloc[0] == "Hindi"


def test_export_comparison_and_ratings_to_csv(sample_comparison_payload):
    """Verify that export_comparison_and_ratings_to_csv creates a valid CSV string."""
    ratings_dict = {
        "marian": {
            "meaning_preservation": 5.0,
            "grammar": 5.0,
            "fluency": 4.5,
            "naturalness": 4.0,
            "overall_usefulness": 4.5,
        },
        "qwen_tone_controlled": {
            "meaning_preservation": 4.5,
            "grammar": 5.0,
            "fluency": 5.0,
            "naturalness": 4.5,
            "overall_usefulness": 5.0,
        },
    }

    csv_output = export_comparison_and_ratings_to_csv(
        sample_comparison_payload,
        ratings_dict=ratings_dict,
    )
    assert isinstance(csv_output, str)
    assert "Original English,Target Language,Model Key" in csv_output
    assert "MarianMT (Helsinki-NLP)" in csv_output
    assert "Qwen2.5-1.5B (Tone: Formal)" in csv_output
    assert "Meaning Preservation (1-5)" in csv_output
    assert "Average Rating" in csv_output


# ============================================================================
# 3. Model Comparison Orchestration Tests (Mocked)
# ============================================================================

def test_run_model_comparison_empty_input_rejection():
    """Verify that empty inputs raise a ValueError."""
    with pytest.raises(ValueError, match="empty or contain only whitespace"):
        run_model_comparison(text="", target_language="Hindi")


@patch("modules.evaluation.translate_text")
@patch("modules.evaluation.translate_with_llm")
def test_run_model_comparison_orchestrates_all_four_candidates(mock_llm, mock_marian):
    """Verify that run_model_comparison properly coordinates all 4 candidate pipelines."""
    # Mock MarianMT response
    mock_marian.return_value = {
        "success": True,
        "original_text": "How are you?",
        "translated_text": "आप कैसे हैं?",
        "target_language": "Hindi",
        "model_name": "Helsinki-NLP/opus-mt-en-hi",
        "device": "cuda",
        "latency_seconds": 0.05,
        "error": None,
    }

    # Mock Qwen response
    mock_llm.return_value = {
        "success": True,
        "original_text": "How are you?",
        "translated_text": "आप कैसे हैं?",
        "target_language": "Hindi",
        "technique": "zero-shot",
        "tone": "neutral",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "device": "cuda",
        "latency_seconds": 0.25,
        "error": None,
    }

    result = run_model_comparison(
        text="How are you?",
        target_language="Hindi",
        selected_tone="formal",
    )

    assert result["original_text"] == "How are you?"
    assert result["target_language"] == "Hindi"
    assert result["selected_tone"] == "formal"
    assert len(result["candidates"]) == 4

    keys = [c["key"] for c in result["candidates"]]
    assert "marian" in keys
    assert "qwen_zero_shot" in keys
    assert "qwen_few_shot" in keys
    assert "qwen_tone_controlled" in keys

    # Verify MarianMT was called once
    mock_marian.assert_called_once_with(text="How are you?", target_language="Hindi")

    # Verify Qwen LLM was called 3 times (zero-shot, few-shot, tone-controlled)
    assert mock_llm.call_count == 3


def test_academic_disclaimer_notice_presence():
    """Verify academic disclaimer text contains objective caveat language."""
    assert "subjective" in DISCLAIMER_TEXT.lower()
    assert "superiority" in DISCLAIMER_TEXT.lower()


# ============================================================================
# 4. Automated Metric (BLEU / chrF) Tests
# ============================================================================

def test_calculate_bleu_chrf_valid():
    """Verify corpus BLEU and chrF scoring using sacrebleu."""
    hyps = ["नमस्ते दुनिया आपका स्वागत है", "शिक्षा बहुत ही महत्वपूर्ण है"]
    refs = [["नमस्ते दुनिया आपका स्वागत है"], ["शिक्षा बहुत महत्वपूर्ण है जीवन में"]]

    metrics = calculate_bleu_chrf(hyps, refs)
    assert metrics["bleu"] is not None
    assert metrics["chrf"] is not None
    assert metrics["bleu"] > 0
    assert metrics["chrf"] > 0
    assert metrics["sample_count"] == 2


def test_calculate_bleu_chrf_missing_references():
    """Verify missing or pending references return None without fabrication."""
    hyps = ["नमस्ते", "धन्यवाद"]
    refs = [[""], ["Pending"]]

    metrics = calculate_bleu_chrf(hyps, refs)
    assert metrics["bleu"] is None
    assert metrics["chrf"] is None
    assert "Pending" in metrics["status"]


def test_compute_sentence_metrics_valid_and_pending():
    """Verify sentence-level BLEU/chrF calculation and pending status."""
    valid_res = compute_sentence_metrics("सुप्रभात", "सुप्रभात")
    assert valid_res["bleu"] is not None
    assert valid_res["chrf"] is not None
    assert valid_res["bleu"] == 100.0

    pending_res = compute_sentence_metrics("सुप्रभात", "")
    assert pending_res["bleu"] is None
    assert "Pending" in pending_res["status"]


def test_load_benchmark_dataset():
    """Verify loading benchmark dataset from data/benchmark_dataset.csv."""
    df = load_benchmark_dataset()
    assert len(df) == 20
    assert "english" in df.columns
    assert "reference_hindi" in df.columns
    assert "reference_marathi" in df.columns


def test_run_benchmark_evaluation_with_mock(tmp_path):
    """Verify benchmark evaluation run with mock models and CSV outputs."""
    sample_data = pd.DataFrame({
        "id": ["BM01", "BM02"],
        "english": ["Hello world", "Good morning"],
        "reference_hindi": ["नमस्ते दुनिया", "सुप्रभात"],
        "reference_marathi": ["नमस्ते जग", "शुभ सकाळ"],
    })

    mock_models = {
        "MockMarian": lambda text, lang: {"success": True, "translated_text": "नमस्ते दुनिया"},
        "MockQwen": lambda text, lang: {"success": True, "translated_text": "सुप्रभात"},
    }

    eval_result = run_benchmark_evaluation(sample_data, "Hindi", mock_models)
    assert "results" in eval_result
    assert "summary" in eval_result
    assert len(eval_result["results"]) == 4  # 2 sentences * 2 models

    saved_files = export_benchmark_outputs(
        hindi_eval=eval_result,
        output_dir=tmp_path,
    )
    assert "hindi_model_comparison" in saved_files
    assert "translation_quality_metrics" in saved_files
    assert "complete_translation_comparison" in saved_files

