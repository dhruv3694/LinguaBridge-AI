"""Standalone unit test runner for LinguaBridge AI."""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_bias_evaluation as tb

test_funcs = [
    tb.test_gender_marker_dictionaries_structure,
    tb.test_hindi_masculine_detection,
    tb.test_hindi_feminine_detection,
    tb.test_marathi_masculine_detection,
    tb.test_marathi_feminine_detection,
    tb.test_mixed_markers,
    tb.test_no_selected_markers,
    tb.test_invalid_language_raises_value_error,
    tb.test_non_string_text_handled_gracefully,
    tb.test_run_bias_evaluation_with_mock,
    tb.test_run_bias_evaluation_missing_columns,
    tb.test_summarize_and_neutral_profession_comparison,
    tb.test_create_gender_pair_and_review_worksheet,
    tb.test_gender_bias_evaluator_class,
]

passed = 0
failed = 0

print(f"Running {len(test_funcs)} unit tests from test_bias_evaluation.py...\n")

for fn in test_funcs:
    try:
        fn()
        print(f"PASS: {fn.__name__}")
        passed += 1
    except Exception as exc:
        print(f"FAIL: {fn.__name__} - {exc}")
        traceback.print_exc()
        failed += 1

print(f"\nResults: {passed} passed, {failed} failed out of {len(test_funcs)} tests.")
if failed > 0:
    sys.exit(1)
print("\nALL BIAS EVALUATION UNIT TESTS PASSED!")
