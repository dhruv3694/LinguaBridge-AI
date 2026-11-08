"""Gender bias evaluation (notebook cell 137 + workflow)."""
from __future__ import annotations
import streamlit as st
import pandas as pd
from modules.bias_evaluation import (
    compare_neutral_professions, create_gender_pair_comparison, create_human_review_worksheet,
    export_bias_outputs, load_bias_dataset, run_bias_evaluation, summarize_bias_results,
)
from modules.translator import translate_text as marian_translate
from modules.llm_translator import translate_with_llm


def render():
    st.title("Gender Bias Evaluation (U5)")
    st.markdown(
        "Heuristic screening of selected Hindi/Marathi grammatical markers in profession translations. "
        "**Marker detection does not independently prove bias**; human linguistic review is required."
    )
    try:
        df = load_bias_dataset()
    except Exception as e:
        st.error(f"Dataset load failed: {e}")
        return
    st.dataframe(df, use_container_width=True)
    languages = st.multiselect("Languages", ["Hindi", "Marathi"], default=["Hindi", "Marathi"])
    models = st.multiselect("Models", ["MarianMT", "Qwen2.5-1.5B"], default=["MarianMT", "Qwen2.5-1.5B"])
    max_rows = st.slider("Dataset rows to evaluate (subset for speed)", 1, len(df), min(10, len(df)))
    if st.button("Run bias evaluation", type="primary", disabled=not (languages and models)):
        funcs = {}
        if "MarianMT" in models:
            funcs["MarianMT"] = lambda t, lang: marian_translate(t, lang)
        if "Qwen2.5-1.5B" in models:
            funcs["Qwen2.5-1.5B"] = lambda t, lang: translate_with_llm(t, target_language=lang, style="Neutral", technique="zero_shot")
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
        paths = export_bias_outputs(results)
        st.success(f"Exported bias outputs to `outputs/bias/`")
        st.download_button("Download human review worksheet (CSV)", review.to_csv(index=False).encode("utf-8-sig"), "gender_bias_human_review.csv")
