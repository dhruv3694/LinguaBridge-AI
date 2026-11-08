"""Text-to-Speech using ai4bharat/indic-parler-tts (notebook)."""
from __future__ import annotations
from pathlib import Path
import streamlit as st
from modules.speech import SUPPORTED_SPEAKERS, VOICE_STYLES, generate_speech
from modules.translator import translate_text as marian_translate
from modules.llm_translator import translate_with_llm
from modules.translation import translate_text as indictrans2_translate


def render():
    st.title("Text-to-Speech (Indic Parler-TTS)")
    st.caption("Notebook model: ai4bharat/indic-parler-tts. Generates local WAV audio for Hindi/Marathi.")
    mode = st.radio("Input mode", ["Direct Hindi/Marathi text", "Translate English then speak"], horizontal=True)
    language = st.selectbox("Language / voice language", ["Hindi", "Marathi"], key="tts_lang")
    speaker = st.selectbox("Speaker", SUPPORTED_SPEAKERS[language])
    style = st.selectbox("Style", list(VOICE_STYLES.keys()))
    text = ""
    if mode.startswith("Direct"):
        text = st.text_area("Text to speak", height=120)
    else:
        eng = st.text_area("English text", height=100)
        engine = st.selectbox("Translator", ["MarianMT", "Qwen2.5", "IndicTrans2 (experimental)"])
        if eng and st.button("Translate for speech"):
            if engine == "MarianMT":
                r = marian_translate(eng, language)
            elif engine == "Qwen2.5":
                r = translate_with_llm(eng, target_language=language)
            else:
                r = indictrans2_translate(eng, language)
            if r.get("success"):
                st.session_state["tts_translated"] = r.get("translated") or r.get("translated_text")
            else:
                st.error(r.get("error"))
        text = st.text_area("Translated text to speak", value=st.session_state.get("tts_translated", ""), height=120)
    if st.button("Generate speech", type="primary", disabled=not (text and text.strip())):
        with st.spinner("Loading TTS model and generating audio (first run can be slow)..."):
            res = generate_speech(text, language=language, speaker=speaker, style=style)
        if res.get("success"):
            st.success(f"Generated {res.get('duration_seconds')}s audio on {res.get('device')}")
            audio_path = res["audio_path"]
            st.audio(audio_path)
            with open(audio_path, "rb") as f:
                st.download_button("Download WAV", f, file_name=Path(audio_path).name, mime="audio/wav")
            st.json({k: v for k, v in res.items() if k != "audio_path"})
            st.caption(f"File: {audio_path}")
        else:
            st.error(res.get("error", "TTS failed."))
            st.info("If gated/authentication is required, connect on the Hugging Face Authentication page.")
