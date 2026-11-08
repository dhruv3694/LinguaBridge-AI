"""MarianMT vs Qwen comparison (notebook Module 4 workflow)."""
from __future__ import annotations

import streamlit as st

from modules.evaluation import (
    evaluate_against_references,
    export_comparison_outputs,
    load_benchmark_dataset,
    run_model_comparison,
    summarize_latency,
)
from modules.llm_translator import translate_with_llm
from modules.translator import translate_text as marian_translate


def _fmt_result(df):
    view = df.copy()
    if "success" in view.columns:
        view["success"] = view["success"].map(lambda x: bool(x) if x is not None else False)
    if "error" in view.columns:
        view["error"] = view["error"].fillna("")
    return view


def render():
    st.title("MarianMT vs Qwen Comparison")
    st.caption(
        "Notebook evaluation workflow: benchmark sentences, measured inference latency, BLEU/chrF only when valid references exist. "
        "Model download/load time is not mixed into per-inference latency."
    )
    target = st.selectbox("Target language", ["Hindi", "Marathi"], key="cmp_lang")
    mode = st.radio("Mode", ["Single sentence", "Benchmark dataset"], horizontal=True)

    def marian_fn(text, language):
        return marian_translate(text, language)

    def qwen_fn(text, language):
        return translate_with_llm(text, target_language=language, style="Neutral", technique="zero_shot", max_new_tokens=160)

    funcs = {"MarianMT": marian_fn, "Qwen2.5-1.5B": qwen_fn}

    if mode == "Single sentence":
        text = st.text_area("English sentence", "Artificial intelligence is transforming the world.", key="cmp_text")
        if st.button("Compare models", type="primary", disabled=not text.strip()):
            with st.spinner("Running MarianMT and Qwen (first run may download models)..."):
                rows = run_model_comparison([text], funcs, target)
            st.dataframe(_fmt_result(rows), use_container_width=True)
            fails = rows[~rows["success"].fillna(False).astype(bool)] if "success" in rows.columns else rows.iloc[0:0]
            for _, r in fails.iterrows():
                st.error(f"{r.get('model')}: {r.get('error')}")
    else:
        try:
            df = load_benchmark_dataset()
        except Exception as exc:
            st.error(f"Could not load benchmark dataset: {exc}")
            return
        st.info(
            "Loads `data/translation_benchmark.csv`/benchmark dataset. BLEU/chrF are computed only when reference "
            "columns are valid and non-empty. Failed model calls are recorded with full errors."
        )
        n = st.slider("Sentences to evaluate (subset for speed)", 1, min(20, len(df)), min(5, len(df)))
        if st.button("Run benchmark comparison", type="primary"):
            sentences = df["english"].head(n).tolist()
            with st.spinner(f"Running {n} sentences × 2 models for {target}..."):
                results = run_model_comparison(sentences, funcs, target)
            results = _fmt_result(results)
            total = len(results)
            ok = int(results["success"].sum()) if "success" in results.columns else 0
            failed = total - ok
            lat = summarize_latency(results)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total runs", total)
            c2.metric("Successful", ok)
            c3.metric("Failed", failed)
            c4.metric("Target", target)
            if not lat.empty:
                st.subheader("Latency / success summary")
                st.dataframe(lat, use_container_width=True)
            st.subheader("Detailed results")
            st.dataframe(results, use_container_width=True)
            err_rows = results[~results["success"].astype(bool)]
            if not err_rows.empty:
                st.subheader("Errors (full)")
                for _, r in err_rows.iterrows():
                    st.warning(f"Row {r.get('id')} | {r.get('model')}: {r.get('error')}")
            ref_col = f"reference_{target.lower()}"
            if ref_col not in df.columns:
                st.warning(f"No reference column `{ref_col}` in dataset; BLEU/chrF skipped.")
                metrics = None
            else:
                metrics = evaluate_against_references(results, ref_col, df.head(n))
                st.subheader("Quality metrics (SacreBLEU)")
                st.dataframe(metrics, use_container_width=True)
                if metrics is not None and (metrics["status"].astype(str).str.contains("Reference translations missing", na=False)).all():
                    st.warning("Reference translations missing/invalid for evaluated rows; BLEU/chrF not calculated.")
            if metrics is not None:
                paths = export_comparison_outputs(results, metrics, target)
                st.success(f"Exported: {paths}")
                st.download_button("Download results CSV", results.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{target.lower()}_comparison_results.csv", mime="text/csv")
