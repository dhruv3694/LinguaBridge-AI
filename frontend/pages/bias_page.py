"""Gender Bias Evaluation with both dataset and user-input workflows."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.bias_evaluation import (
    compare_neutral_professions,
    create_gender_pair_comparison,
    create_human_review_worksheet,
    evaluate_user_input,
    export_bias_outputs,
    load_bias_dataset,
    load_bias_history,
    run_bias_evaluation,
    summarize_bias_results,
)
from modules.llm_translator import translate_with_llm
from modules.translator import translate_text as marian_translate


def _build_model_mapping(models):
    mapping = {}
    if "MarianMT" in models:
        mapping["MarianMT"] = lambda text, lang: marian_translate(text, lang)
    if "Qwen2.5-1.5B" in models:
        mapping["Qwen2.5-1.5B"] = lambda text, lang: translate_with_llm(text, target_language=lang, style="Neutral", technique="zero_shot")
    return mapping


def _render_user_input_mode():
    st.subheader("Mode 1: User Input Evaluation")
    st.caption("Enter an English sentence, select a target language and translation model, then evaluate the actual generated Hindi/Marathi output.")

    english_input = st.text_area("English sentence", height=150, key="bias_user_text")
    target_language = st.selectbox("Target language", ["Hindi", "Marathi"], key="bias_user_lang")
    translation_model = st.selectbox("Translation model", ["MarianMT", "Qwen2.5-1.5B"], key="bias_user_model")

    if st.button("Translate and Evaluate Gender Bias", type="primary", key="bias_user_submit"):
        if not english_input or not english_input.strip():
            st.error("Please enter a valid English sentence.")
            return
        model_map = _build_model_mapping([translation_model])
        fn = model_map[translation_model]
        status = st.status("Translating and screening gender markers...")
        with status:
            result = evaluate_user_input(english_input, target_language, translation_model, fn)
        if not result.get("success"):
            st.error(result.get("error", "Translation failed."))
            return

        st.success("Evaluation completed successfully.")
        st.subheader("Evaluation Results")
        st.json({k: v for k, v in result.items() if k not in {"csv_path"}})

        st.divider()
        history = load_bias_history()
        if not history.empty:
            st.subheader("Latest evaluation")
            latest = history.iloc[-1]
            st.dataframe(latest.to_frame().T, use_container_width=True)
            st.subheader("History")
            st.dataframe(history, use_container_width=True)
            st.download_button(
                "Download evaluation history (CSV)",
                history.to_csv(index=False).encode("utf-8-sig"),
                file_name="user_input_bias_evaluations.csv",
                mime="text/csv",
                key="bias_history_download",
            )


def _render_dataset_mode():
    st.subheader("Mode 2: Dataset Evaluation")
    try:
        df = load_bias_dataset()
    except Exception as exc:
        st.error(f"Dataset load failed: {exc}")
        return

    st.dataframe(df, use_container_width=True)
    languages = st.multiselect("Languages", ["Hindi", "Marathi"], default=["Hindi", "Marathi"])
    models = st.multiselect("Models", ["MarianMT", "Qwen2.5-1.5B"], default=["MarianMT", "Qwen2.5-1.5B"])
    max_rows = st.slider("Dataset rows to evaluate (subset for speed)", 1, len(df), min(10, len(df)))
    if st.button("Run dataset bias evaluation", type="primary", disabled=not (languages and models), key="bias_dataset_submit"):
        funcs = _build_model_mapping(models)
        if not funcs:
            st.error("Select at least one model for evaluation.")
            return
        subset = df.head(max_rows)
        all_results = []
        with st.spinner("Translating and screening markers..."):
            for lang in languages:
                for name, fn in funcs.items():
                    all_results.append(run_bias_evaluation(subset, fn, lang, name))
        results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        st.session_state["bias_results"] = results
        st.subheader("Results")
        st.dataframe(results, use_container_width=True)
        st.subheader("Summary")
        st.dataframe(summarize_bias_results(results), use_container_width=True)
        st.subheader("Neutral profession comparison")
        st.dataframe(compare_neutral_professions(results), use_container_width=True)
        st.subheader("Gender-pair comparison")
        st.dataframe(create_gender_pair_comparison(results), use_container_width=True)
        review = create_human_review_worksheet(results)
        export_bias_outputs(results)
        st.success("Dataset bias outputs exported to outputs/bias/")
        st.download_button("Download human review worksheet (CSV)", review.to_csv(index=False).encode("utf-8-sig"), "gender_bias_human_review.csv")


def render():
    st.title("Gender Bias Evaluation (U5)")
    st.markdown(
        "Heuristic screening of Hindi/Marathi grammatical markers. This screens for possible gender assumptions using the actual output of the selected translation model; it does not prove model bias on its own."
    )
    mode = st.radio("Select evaluation mode", ["User Input Evaluation", "Dataset Evaluation"], horizontal=True, key="bias_mode")
    if mode == "User Input Evaluation":
        _render_user_input_mode()
    else:
        _render_dataset_mode()
