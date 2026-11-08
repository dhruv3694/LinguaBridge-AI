"""Evaluation engine (notebook cell 43)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional


def _paths():
    from config.settings import DATA_DIR, OUTPUTS_DIR

    return DATA_DIR, OUTPUTS_DIR


def calculate_metrics(predictions, references):
    import sacrebleu

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal lengths.")
    if not predictions:
        raise ValueError("No translations provided.")
    if not all(str(ref).strip() for ref in references):
        raise ValueError("All reference translations must be populated.")
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {"BLEU": round(bleu.score, 3), "chrF": round(chrf.score, 3)}


def _normalize_result(output: Any, model_name: str, elapsed: Optional[float]) -> Dict[str, Any]:
    if isinstance(output, dict):
        success = bool(output.get("success"))
        translated = output.get("translated") or output.get("translated_text") or ""
        if not success:
            translated = "" if translated is None else translated
        error = output.get("error")
        latency = output.get("latency_seconds")
        if latency is None and not success:
            latency = None
        elif latency is None:
            latency = elapsed
        device = output.get("device")
    else:
        success = bool(output)
        translated = str(output) if output else ""
        error = None if success else "Translation function returned no output."
        latency = elapsed
        device = None
    return {
        "success": success,
        "translation": translated if success else (translated or ""),
        "latency_seconds": latency,
        "error": error,
        "device": device,
    }


def run_model_comparison(sentences, translation_functions, target_language):
    import pandas as pd

    results = []
    for sentence_id, sentence in enumerate(sentences, start=1):
        for model_name, translate_function in translation_functions.items():
            start_time = time.perf_counter()
            try:
                output = translate_function(sentence, target_language)
                elapsed = round(time.perf_counter() - start_time, 4)
                norm = _normalize_result(output, model_name, elapsed)
                results.append(
                    {
                        "id": sentence_id,
                        "english": sentence,
                        "target_language": target_language,
                        "model": model_name,
                        "translation": norm["translation"],
                        "latency_seconds": norm["latency_seconds"],
                        "device": norm.get("device"),
                        "success": norm["success"],
                        "error": norm["error"],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "id": sentence_id,
                        "english": sentence,
                        "target_language": target_language,
                        "model": model_name,
                        "translation": "",
                        "latency_seconds": None,
                        "device": None,
                        "success": False,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
    return pd.DataFrame(results)


def summarize_latency(results):
    import pandas as pd

    if results.empty:
        return pd.DataFrame()
    df = results.copy()
    df["success"] = df["success"].fillna(False).astype(bool)
    df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce")
    summary = (
        df.groupby("model")
        .agg(
            total=("success", "count"),
            successful=("success", "sum"),
            mean_latency_s=("latency_seconds", "mean"),
            median_latency_s=("latency_seconds", "median"),
            min_latency_s=("latency_seconds", "min"),
            max_latency_s=("latency_seconds", "max"),
        )
        .reset_index()
    )
    summary["success_rate_pct"] = (summary["successful"] / summary["total"] * 100).round(1)
    for c in ["mean_latency_s", "median_latency_s", "min_latency_s", "max_latency_s"]:
        summary[c] = summary[c].round(4)
    return summary


def evaluate_against_references(results, reference_column, benchmark_df):
    import pandas as pd

    merged = results.merge(benchmark_df[["id", reference_column]], on="id", how="left")
    summaries = []
    for model_name, group in merged.groupby("model"):
        valid = group[
            (group["success"] == True)
            & group[reference_column].notna()
            & (group[reference_column].astype(str).str.strip() != "")
        ]
        if valid.empty:
            summaries.append(
                {
                    "model": model_name,
                    "BLEU": None,
                    "chrF": None,
                    "evaluated_sentences": 0,
                    "status": "Reference translations missing or no successful translations",
                }
            )
            continue
        try:
            scores = calculate_metrics(
                valid["translation"].astype(str).tolist(),
                valid[reference_column].astype(str).tolist(),
            )
            summaries.append({"model": model_name, **scores, "evaluated_sentences": len(valid), "status": "Calculated"})
        except Exception as exc:
            summaries.append(
                {
                    "model": model_name,
                    "BLEU": None,
                    "chrF": None,
                    "evaluated_sentences": len(valid),
                    "status": f"Metric error: {exc}",
                }
            )
    return pd.DataFrame(summaries)


def load_benchmark_dataset(path: Optional[Path] = None):
    import pandas as pd

    DATA_DIR, _ = _paths()
    for p in [path, DATA_DIR / "translation_benchmark.csv", DATA_DIR / "benchmark_dataset.csv"]:
        if p and Path(p).exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Benchmark dataset not found in data/.")


def export_comparison_outputs(results, metrics, language: str) -> Dict[str, str]:
    _, OUTPUTS_DIR = _paths()

    out = OUTPUTS_DIR / "evaluations"
    out.mkdir(parents=True, exist_ok=True)
    comp = out / f"{language.lower()}_model_comparison.csv"
    met = out / f"{language.lower()}_quality_metrics.csv"
    lat = out / f"{language.lower()}_latency_summary.csv"
    results.to_csv(comp, index=False, encoding="utf-8-sig")
    metrics.to_csv(met, index=False, encoding="utf-8-sig")
    try:
        summarize_latency(results).to_csv(lat, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return {"comparison": str(comp), "metrics": str(met), "latency": str(lat)}
