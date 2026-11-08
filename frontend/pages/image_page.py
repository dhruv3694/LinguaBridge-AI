"""Image captioning and translation (BLIP + selectable translator)."""
from __future__ import annotations
from pathlib import Path
import tempfile
import streamlit as st
from modules.image_caption import caption_image
from modules.image_translation import process_image
from modules.translator import translate_text as marian_translate
from modules.llm_translator import translate_with_llm


def render():
    st.title("Image Captioning & Translation")
    st.caption("Notebook pipeline: BLIP generates English caption; selected translator produces Hindi/Marathi.")
    target = st.selectbox("Target language", ["Hindi", "Marathi"], key="img_lang")
    engine = st.selectbox("Translation engine for caption", ["MarianMT", "Qwen2.5"])
    up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if up and st.button("Generate caption & translate", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
            tmp.write(up.getbuffer())
            tmp_path = tmp.name
        st.image(up, caption="Input image", use_container_width=True)
        def fn(text, language):
            if engine == "MarianMT":
                return marian_translate(text, language)
            return translate_with_llm(text, target_language=language, style="Neutral", technique="zero_shot")
        with st.spinner("Running BLIP and translation..."):
            res = process_image(tmp_path, target, fn)
        if res.get("success"):
            st.subheader("English caption")
            st.write(res.get("english_caption"))
            st.subheader(f"{target} translation")
            st.write(res.get("translated_caption"))
            st.json({k: v for k, v in res.items() if k not in ("english_caption", "translated_caption")})
        else:
            st.error(f"Stage: {res.get('stage','?')} — {res.get('error')}")
            cap = caption_image(tmp_path)
            if cap.get("success"):
                st.info(f"Caption (before translation failure): {cap.get('caption')}")
