"""
LinguaBridge AI - Unit Tests for Gender Bias Evaluation Module
Phase 7: Occupational Gender Bias Screening
"""

import pandas as pd
import pytest

from modules.bias_evaluation import (
    GENDER_MARKERS,
    GenderBiasEvaluator,
    compare_neutral_professions,
    create_gender_pair_comparison,
    create_human_review_worksheet,
    detect_gender_markers,
    run_bias_evaluation,
    summarize_bias_results,
)


def test_gender_marker_dictionaries_structure():
    """Verify that marker dictionaries exist for Hindi and Marathi with correct keys."""
    assert "Hindi" in GENDER_MARKERS
    assert "Marathi" in GENDER_MARKERS
    for lang in ["Hindi", "Marathi"]:
        assert "masculine" in GENDER_MARKERS[lang]
        assert "feminine" in GENDER_MARKERS[lang]
        assert len(GENDER_MARKERS[lang]["masculine"]) > 0
        assert len(GENDER_MARKERS[lang]["feminine"]) > 0


def test_hindi_masculine_detection():
    """Test detection of masculine markers in Hindi."""
    res = detect_gender_markers("वह स्कूल जाता है और अच्छा लिखता है।", "Hindi")
    assert res["classification"] == "masculine_markers_detected"
    assert "लिखता" in res["masculine_markers"]
    assert res["masculine_count"] == 1
    assert res["feminine_count"] == 0


def test_hindi_feminine_detection():
    """Test detection of feminine markers in Hindi."""
    res = detect_gender_markers("वह शिक्षिका अच्छा लिखती है।", "Hindi")
    assert res["classification"] == "feminine_markers_detected"
    assert "लिखती" in res["feminine_markers"]
    assert res["masculine_count"] == 0
    assert res["feminine_count"] == 1


def test_marathi_masculine_detection():
    """Test detection of masculine markers in Marathi."""
    res = detect_gender_markers("तो शाळेत जातो आणि लिहितो.", "Marathi")
    assert res["classification"] == "masculine_markers_detected"
    assert "लिहितो" in res["masculine_markers"]
    assert res["masculine_count"] == 1
    assert res["feminine_count"] == 0


def test_marathi_feminine_detection():
    """Test detection of feminine markers in Marathi."""
    res = detect_gender_markers("ती काम करते आणि पत्र लिहिते.", "Marathi")
    assert res["classification"] == "feminine_markers_detected"
    assert "करते" in res["feminine_markers"] or "लिहिते" in res["feminine_markers"]
    assert res["feminine_count"] >= 1
    assert res["masculine_count"] == 0


def test_mixed_markers():
    """Test classification when both masculine and feminine markers are present."""
    # "लिखता" (masc) and "गई" (fem) in Hindi
    res = detect_gender_markers("वह लिखता था और वह गई।", "Hindi")
    assert res["classification"] == "mixed_markers"
    assert res["masculine_count"] >= 1
    assert res["feminine_count"] >= 1


def test_no_selected_markers():
    """Test classification when no markers from the heuristic list appear."""
    res = detect_gender_markers("नमस्ते, आप कैसे हैं?", "Hindi")
    assert res["classification"] == "no_selected_marker_detected"
    assert res["masculine_count"] == 0
    assert res["feminine_count"] == 0


def test_invalid_language_raises_value_error():
    """Invalid language parameter must raise a ValueError."""
    with pytest.raises(ValueError, match="Language must be Hindi or Marathi"):
        detect_gender_markers("Hello", "German")


def test_non_string_text_handled_gracefully():
    """None, numbers or other non-string values should be handled safely."""
    res = detect_gender_markers(None, "Hindi")
    assert res["classification"] == "no_selected_marker_detected"
    assert res["masculine_count"] == 0


def test_run_bias_evaluation_with_mock():
    """Test end-to-end bias evaluation run on a sample dataset."""
    sample_df = pd.DataFrame({
        "id": [1, 2],
        "gender_condition": ["neutral", "masculine"],
        "profession": ["Doctor", "Engineer"],
        "english": ["The doctor is working.", "He is an engineer."],
    })

    def mock_translator(text, lang):
        if "doctor" in text.lower():
            return {"success": True, "translated": "डॉक्टर काम करता है।"}
        return {"success": True, "translated": "वह इंजीनियर गया।"}

    results = run_bias_evaluation(
        dataset=sample_df,
        translation_function=mock_translator,
        target_language="Hindi",
        model_name="MockModel",
    )

    assert len(results) == 2
    assert "marker_classification" in results.columns
    assert results.iloc[0]["marker_classification"] == "masculine_markers_detected"
    assert results.iloc[0]["masculine_markers"] == "करता"
    assert results.iloc[1]["masculine_markers"] == "गया"


def test_run_bias_evaluation_missing_columns():
    """Missing required dataset columns must raise a ValueError."""
    bad_df = pd.DataFrame({"id": [1], "english": ["Test"]})

    with pytest.raises(ValueError, match="Missing dataset columns"):
        run_bias_evaluation(bad_df, lambda t, l: "", "Hindi", "MockModel")


def test_summarize_and_neutral_profession_comparison():
    """Test group aggregation helpers on bias evaluation results."""
    results_df = pd.DataFrame({
        "id": [1, 2, 3],
        "gender_condition": ["neutral", "neutral", "masculine"],
        "profession": ["Doctor", "Nurse", "Doctor"],
        "english": ["e1", "e2", "e3"],
        "target_language": ["Hindi", "Hindi", "Hindi"],
        "model": ["Mock", "Mock", "Mock"],
        "translation": ["t1", "t2", "t3"],
        "success": [True, True, True],
        "marker_classification": [
            "masculine_markers_detected",
            "feminine_markers_detected",
            "masculine_markers_detected",
        ],
        "masculine_markers": ["करता", "", "करता"],
        "feminine_markers": ["", "करती", ""],
        "error": ["", "", ""],
        "human_review": ["Pending", "Pending", "Pending"],
    })

    summary = summarize_bias_results(results_df)
    assert not summary.empty
    assert "count" in summary.columns

    neutral_comp = compare_neutral_professions(results_df)
    assert not neutral_comp.empty
    assert len(neutral_comp) == 2  # Doctor and Nurse neutral


def test_create_gender_pair_and_review_worksheet():
    """Test paired comparison pivot table and human review worksheet generation."""
    paired_df = pd.DataFrame({
        "id": [1, 2],
        "gender_condition": ["masculine", "feminine"],
        "profession": ["Doctor", "Doctor"],
        "english": ["He is a doctor.", "She is a doctor."],
        "target_language": ["Hindi", "Hindi"],
        "model": ["Mock", "Mock"],
        "translation": ["वह डॉक्टर गया।", "वह डॉक्टर गई।"],
        "success": [True, True],
        "marker_classification": ["masculine_markers_detected", "feminine_markers_detected"],
        "masculine_markers": ["गया", ""],
        "feminine_markers": ["", "गई"],
        "error": ["", ""],
        "human_review": ["Pending", "Pending"],
    })

    pair_pivot = create_gender_pair_comparison(paired_df)
    assert not pair_pivot.empty
    assert "masculine" in pair_pivot.columns
    assert "feminine" in pair_pivot.columns

    review_sheet = create_human_review_worksheet(paired_df)
    assert "meaning_preserved" in review_sheet.columns
    assert "reviewer_comments" in review_sheet.columns


def test_gender_bias_evaluator_class():
    """Verify OOP GenderBiasEvaluator load and pair evaluation."""
    evaluator = GenderBiasEvaluator()
    assert evaluator.professions_data is not None
    assert len(evaluator.professions_data) == 20

    res = evaluator.evaluate_pair(
        source_text="The doctor is working.",
        marian_translation="डॉक्टर काम करता है।",
        llm_translation="डॉक्टर काम करती है।",
        language="Hindi",
    )
    assert res["marian"]["classification"] == "masculine_markers_detected"
    assert res["llm"]["classification"] == "feminine_markers_detected"
