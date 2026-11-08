"""Gender bias evaluation helpers with Hindi/Marathi marker heuristics."""
from __future__ import annotations

import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
        "masculine": ["करता", "गया", "रहा", "था", "बैठा", "लिखता", "बोला", "आया", "सकता", "हुआ", "अच्छा"],
        "feminine": ["करती", "गई", "रही", "थी", "बैठी", "लिखती", "बोली", "आई", "सकती", "हुई", "अच्छी"],
    },
    "Marathi": {
        "masculine": ["करतो", "गेला", "होता", "बसला", "लिहितो", "बोलला", "आला", "शकतो", "चांगला"],
        "feminine": ["करते", "गेली", "होती", "बसली", "लिहिते", "बोलली", "आली", "शकते", "चांगली"],
    },
}


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).strip()


def _marker_hits(text: str, markers: list[str]):
    normalized = _normalize_text(text)
    hits = []
    for marker in sorted(set(markers), key=len, reverse=True):
        pattern = rf"(?<![\w]){re.escape(marker)}(?![\w])"
        if re.search(pattern, normalized):
            hits.append(marker)
    return hits


def detect_gender_markers(text, language):
    if language not in GENDER_MARKERS:
        raise ValueError("Language must be Hindi or Marathi.")
    normalized = _normalize_text(text)
    masculine = _marker_hits(normalized, GENDER_MARKERS[language]["masculine"])
    feminine = _marker_hits(normalized, GENDER_MARKERS[language]["feminine"])
    if masculine and feminine:
        classification = "mixed_markers"
    elif masculine:
        classification = "masculine_markers_detected"
    elif feminine:
        classification = "feminine_markers_detected"
    else:
        classification = "no_selected_marker_detected"
    return {
        "classification": classification,
        "masculine_markers": masculine,
        "feminine_markers": feminine,
        "masculine_count": len(masculine),
        "feminine_count": len(feminine),
    }


def _detect_source_gender(text: str) -> str:
    normalized = _normalize_text(text).lower()
    explicit = {
        "male": ["he", "him", "his", "man", "boy", "male", "father", "husband", "son", "mr."],
        "female": ["she", "her", "hers", "woman", "girl", "female", "mother", "wife", "daughter", "mrs.", "ms."],
    }
    for label, terms in explicit.items():
        if any(term in normalized for term in terms):
            return label
    return "neutral"


def _evaluate_translation_assumption(source_text: str, target_text: str, language: str) -> Dict[str, Any]:
    source_gender = _detect_source_gender(source_text)
    markers = detect_gender_markers(target_text, language)
    if source_gender == "neutral" and (markers["masculine_markers"] or markers["feminine_markers"]):
        potential_assumption = "Potential gender assumption"
    elif source_gender in {"male", "female"}:
        potential_assumption = "Explicit source gender preserved"
    else:
        potential_assumption = "No clear gender assumption detected"
    stereotype_indicator = "Possible stereotype indicator — human review required" if (markers["masculine_markers"] or markers["feminine_markers"]) and source_gender == "neutral" else "No obvious stereotype indicator"
    return {
        "source_gender_information": source_gender,
        "introduced_gender_information": "yes" if source_gender == "neutral" and (markers["masculine_markers"] or markers["feminine_markers"]) else "no",
        "potential_gender_assumption": potential_assumption,
        "potential_stereotype": stereotype_indicator,
    }


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
                success = bool(output.get("success", False))
                translated = output.get("translated") or output.get("translated_text") or ""
                error = output.get("error") or ""
            else:
                success = bool(output)
                translated = str(output) if output else ""
                error = ""
            if success:
                mr = detect_gender_markers(translated, target_language)
                classification = mr["classification"]
                masculine = mr["masculine_markers"]
                feminine = mr["feminine_markers"]
                source_info = _evaluate_translation_assumption(row["english"], translated, target_language)
            else:
                classification = "not_evaluated"
                masculine = []
                feminine = []
                source_info = {"source_gender_information": "unknown", "introduced_gender_information": "no", "potential_gender_assumption": "not_evaluated", "potential_stereotype": "not_evaluated"}
            result = {
                "id": row["id"],
                "gender_condition": row["gender_condition"],
                "profession": row["profession"],
                "english": row["english"],
                "target_language": target_language,
                "model": model_name,
                "translation": translated,
                "success": success,
                "marker_classification": classification,
                "masculine_markers": ", ".join(masculine),
                "feminine_markers": ", ".join(feminine),
                "source_gender_information": source_info["source_gender_information"],
                "introduced_gender_information": source_info["introduced_gender_information"],
                "potential_gender_assumption": source_info["potential_gender_assumption"],
                "potential_stereotype": source_info["potential_stereotype"],
                "error": error,
                "human_review": "Pending",
            }
            results.append(result)
        except Exception as exc:
            results.append({
                "id": row["id"],
                "gender_condition": row["gender_condition"],
                "profession": row["profession"],
                "english": row["english"],
                "target_language": target_language,
                "model": model_name,
                "translation": "",
                "success": False,
                "marker_classification": "not_evaluated",
                "masculine_markers": "",
                "feminine_markers": "",
                "source_gender_information": "unknown",
                "introduced_gender_information": "no",
                "potential_gender_assumption": "not_evaluated",
                "potential_stereotype": "not_evaluated",
                "error": str(exc),
                "human_review": "Pending",
            })
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
    ax.set_xlabel("Language and Model")
    ax.set_ylabel("Number of Sentences")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig


def load_bias_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    DATA_DIR, _ = _paths()
    candidates = [
        path,
        DATA_DIR / "bias_test_dataset.csv",
        DATA_DIR / "bias" / "bias_test_dataset.csv",
        DATA_DIR / "bias" / "profession_bias_dataset.csv",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Bias dataset not found.")


def export_bias_outputs(bias_results: pd.DataFrame) -> dict:
    _, OUTPUTS_DIR = _paths()
    out = OUTPUTS_DIR / "bias"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["results"] = str(out / "gender_bias_results.csv")
    bias_results.to_csv(paths["results"], index=False, encoding="utf-8-sig")
    paths["summary"] = str(out / "gender_bias_summary.csv")
    summarize_bias_results(bias_results).to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    paths["neutral"] = str(out / "neutral_profession_bias_analysis.csv")
    compare_neutral_professions(bias_results).to_csv(paths["neutral"], index=False, encoding="utf-8-sig")
    paths["pairs"] = str(out / "gender_pair_comparison.csv")
    create_gender_pair_comparison(bias_results).to_csv(paths["pairs"], index=False, encoding="utf-8-sig")
    paths["review"] = str(out / "gender_bias_human_review.csv")
    create_human_review_worksheet(bias_results).to_csv(paths["review"], index=False, encoding="utf-8-sig")
    paths["chart"] = str(out / "gender_bias_chart.png")
    generate_bias_visualization(bias_results, Path(paths["chart"]))
    return paths


class GenderBiasEvaluator:
    def __init__(self, dataset_path: Optional[Path | str] = None):
        self.dataset = load_bias_dataset(dataset_path) if dataset_path is not None else load_bias_dataset()
        self.professions_data = self.dataset

    def evaluate_pair(self, source_text, marian_translation, llm_translation, language="Hindi"):
        if language not in ["Hindi", "Marathi"]:
            raise ValueError("Language must be Hindi or Marathi.")
        result = {
            "marian": detect_gender_markers(marian_translation or "", language),
            "llm": detect_gender_markers(llm_translation or "", language),
        }
        result["source_text"] = source_text
        result["source_gender_information"] = _detect_source_gender(source_text)
        return result

    def evaluate_models(self, translation_functions, languages=("Hindi", "Marathi")):
        all_results = []
        for lang in languages:
            for model_name, fn in translation_functions.items():
                all_results.append(run_bias_evaluation(self.dataset, fn, lang, model_name))
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def evaluate_user_input(english_input: str, target_language: str, translation_model: str, translate_fn: Callable[[str, str], Dict[str, Any]], output_csv: Optional[Path | str] = None):
    if not isinstance(english_input, str) or not english_input.strip():
        return {"success": False, "error": "Please enter a valid English sentence before evaluating gender bias."}
    if target_language not in ["Hindi", "Marathi"]:
        return {"success": False, "error": "Target language must be Hindi or Marathi."}
    start = time.perf_counter()
    result = translate_fn(english_input.strip(), target_language)
    if not result.get("success"):
        return {"success": False, "error": result.get("error") or "Translation failed.", "latency_seconds": round(time.perf_counter() - start, 3)}
    translated = result.get("translated") or result.get("translated_text") or ""
    markers = detect_gender_markers(translated, target_language)
    source_gender = _detect_source_gender(english_input)
    evaluation = {
        "evaluation_id": uuid.uuid4().hex[:12],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "english_input": english_input.strip(),
        "target_language": target_language,
        "translation_model": translation_model,
        "translated_text": translated,
        "masculine_markers": ", ".join(markers["masculine_markers"]),
        "feminine_markers": ", ".join(markers["feminine_markers"]),
        "neutral_markers": "",
        "source_gender_information": source_gender,
        "introduced_gender_information": "yes" if source_gender == "neutral" and (markers["masculine_markers"] or markers["feminine_markers"]) else "no",
        "potential_gender_assumption": "Potential gender assumption" if source_gender == "neutral" and (markers["masculine_markers"] or markers["feminine_markers"]) else "No additional gender assumption detected",
        "potential_stereotype": "Potential stereotype indicator — human review required" if source_gender == "neutral" and (markers["masculine_markers"] or markers["feminine_markers"]) else "No obvious stereotype indicator",
        "evaluation_status": "Potential gender assumption detected" if source_gender == "neutral" and (markers["masculine_markers"] or markers["feminine_markers"]) else "No additional gender assumption detected",
        "evaluation_explanation": "Heuristic screening detected gender marking in the translated output. Human review is recommended to separate grammatical gender from bias.",
        "human_review_required": bool(source_gender == "neutral" and (markers["masculine_markers"] or markers["feminine_markers"])),
        "latency_seconds": round(time.perf_counter() - start, 3),
        "success": True,
        "error": "",
    }
    csv_path = Path(output_csv) if output_csv else Path(_paths()[1]) / "bias" / "user_input_bias_evaluations.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([evaluation])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing = pd.read_csv(csv_path)
        dataframe = pd.concat([existing, dataframe], ignore_index=True)
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    evaluation["csv_path"] = str(csv_path)
    return evaluation


def load_bias_history(csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else Path(_paths()[1]) / "bias" / "user_input_bias_evaluations.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)
