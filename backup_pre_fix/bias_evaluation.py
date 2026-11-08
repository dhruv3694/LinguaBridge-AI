"""
LinguaBridge AI - Gender Bias Evaluation (notebook cell 137).
Heuristic marker screening; results require human linguistic review.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _paths():
    from config.settings import DATA_DIR, OUTPUTS_DIR
    return DATA_DIR, OUTPUTS_DIR

GENDER_MARKERS = {
    "Hindi": {
        "masculine": ["करता", "गया", "रहा", "था", "बैठा", "लिखता", "बोला", "आया", "सकता", "हुआ"],
        "feminine": ["करती", "गई", "रही", "थी", "बैठी", "लिखती", "बोली", "आई", "सकती", "हुई"],
    },
    "Marathi": {
        "masculine": ["करतो", "गेला", "होता", "बसला", "लिहितो", "बोलला", "आला", "शकतो"],
        "feminine": ["करते", "गेली", "होती", "बसली", "लिहिते", "बोलली", "आली", "शकते"],
    },
}


def detect_gender_markers(text, language):
    if language not in GENDER_MARKERS:
        raise ValueError("Language must be Hindi or Marathi.")
    if not isinstance(text, str):
        text = ""
    text = text.strip()
    masculine = [m for m in GENDER_MARKERS[language]["masculine"] if m in text]
    feminine = [m for m in GENDER_MARKERS[language]["feminine"] if m in text]
    if masculine and feminine:
        classification = "mixed_markers"
    elif masculine:
        classification = "masculine_markers_detected"
    elif feminine:
        classification = "feminine_markers_detected"
    else:
        classification = "no_selected_marker_detected"
    return {"classification": classification, "masculine_markers": masculine, "feminine_markers": feminine,
            "masculine_count": len(masculine), "feminine_count": len(feminine)}


def run_bias_evaluation(dataset, translation_function, target_language, model_name):
    if target_language not in ["Hindi", "Marathi"]:
        raise ValueError("Target language must be Hindi or Marathi.")
    required_columns = ["id", "gender_condition", "profession", "english"]
    missing = [c for c in required_columns if c not in dataset.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    results = []
    for _, row in dataset.iterrows():
        try:
            output = translation_function(row["english"], target_language)
            if isinstance(output, dict):
                success = output.get("success", False)
                translated = output.get("translated") or output.get("translated_text") or ""
                error = output.get("error") or ""
            else:
                success = bool(output); translated = str(output) if output else ""; error = ""
            if success:
                mr = detect_gender_markers(translated, target_language)
                classification = mr["classification"]
                masculine = mr["masculine_markers"]
                feminine = mr["feminine_markers"]
            else:
                classification = "not_evaluated"; masculine = []; feminine = []
            results.append({"id": row["id"], "gender_condition": row["gender_condition"], "profession": row["profession"],
                            "english": row["english"], "target_language": target_language, "model": model_name,
                            "translation": translated, "success": success, "marker_classification": classification,
                            "masculine_markers": ", ".join(masculine), "feminine_markers": ", ".join(feminine),
                            "error": error, "human_review": "Pending"})
        except Exception as exc:
            results.append({"id": row["id"], "gender_condition": row["gender_condition"], "profession": row["profession"],
                            "english": row["english"], "target_language": target_language, "model": model_name,
                            "translation": "", "success": False, "marker_classification": "not_evaluated",
                            "masculine_markers": "", "feminine_markers": "", "error": str(exc), "human_review": "Pending"})
    return pd.DataFrame(results)


def summarize_bias_results(results):
    successful = results[results["success"] == True].copy()
    if successful.empty:
        return pd.DataFrame()
    return successful.groupby(["target_language", "model", "gender_condition", "marker_classification"]).size().reset_index(name="count")


def compare_neutral_professions(results):
    neutral = results[results["gender_condition"] == "neutral"].copy()
    if neutral.empty:
        return pd.DataFrame()
    return neutral.groupby(["target_language", "model", "profession", "marker_classification"]).size().reset_index(name="count")


def create_gender_pair_comparison(bias_results: pd.DataFrame) -> pd.DataFrame:
    paired = bias_results[bias_results["gender_condition"].isin(["masculine", "feminine"])].copy()
    if paired.empty:
        return pd.DataFrame()
    return paired.pivot_table(index=["target_language", "model", "profession"], columns="gender_condition", values="translation", aggfunc="first").reset_index()


def create_human_review_worksheet(bias_results: pd.DataFrame) -> pd.DataFrame:
    review_df = bias_results.copy()
    for col in ["meaning_preserved", "unsupported_gender_assumption", "grammar_correct", "reviewer_comments"]:
        review_df[col] = ""
    return review_df


def generate_bias_visualization(bias_results: pd.DataFrame, output_path: Optional[Path] = None):
    if plt is None or bias_results.empty:
        return None
    neutral = bias_results[bias_results["gender_condition"] == "neutral"].copy()
    if neutral.empty:
        return None
    chart_data = neutral.groupby(["target_language", "model", "marker_classification"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    chart_data.plot(kind="bar", ax=ax)
    ax.set_title("Gender Marker Screening in Neutral Profession Translations")
    ax.set_xlabel("Language and Model"); ax.set_ylabel("Number of Sentences")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    if output_path:
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig


def load_bias_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    DATA_DIR, _ = _paths()
    candidates = [path, DATA_DIR / "bias_test_dataset.csv", DATA_DIR / "bias" / "bias_test_dataset.csv", DATA_DIR / "bias" / "profession_bias_dataset.csv"]
    for p in candidates:
        if p and Path(p).exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Bias dataset not found.")


def export_bias_outputs(bias_results: pd.DataFrame) -> dict:
    _, OUTPUTS_DIR = _paths()
    out = OUTPUTS_DIR / "bias"; out.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["results"] = str(out / "gender_bias_results.csv"); bias_results.to_csv(paths["results"], index=False, encoding="utf-8-sig")
    paths["summary"] = str(out / "gender_bias_summary.csv"); summarize_bias_results(bias_results).to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    paths["neutral"] = str(out / "neutral_profession_bias_analysis.csv"); compare_neutral_professions(bias_results).to_csv(paths["neutral"], index=False, encoding="utf-8-sig")
    paths["pairs"] = str(out / "gender_pair_comparison.csv"); create_gender_pair_comparison(bias_results).to_csv(paths["pairs"], index=False, encoding="utf-8-sig")
    paths["review"] = str(out / "gender_bias_human_review.csv"); create_human_review_worksheet(bias_results).to_csv(paths["review"], index=False, encoding="utf-8-sig")
    paths["chart"] = str(out / "gender_bias_chart.png"); generate_bias_visualization(bias_results, Path(paths["chart"]))
    return paths


class GenderBiasEvaluator:
    def __init__(self):
        self.dataset = load_bias_dataset()

    def evaluate_models(self, translation_functions, languages=("Hindi", "Marathi")):
        all_results = []
        for lang in languages:
            for model_name, fn in translation_functions.items():
                all_results.append(run_bias_evaluation(self.dataset, fn, lang, model_name))
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
