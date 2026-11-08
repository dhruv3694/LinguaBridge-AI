"""Text-to-Speech using ai4bharat/indic-parler-tts with a clearer generation flow."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from modules.speech import SUPPORTED_SPEAKERS, VOICE_STYLES, generate_speech
from modules.translator import translate_text as marian_translate
from modules.llm_translator import translate_with_llm


def render():
    st.title("Text-to-Speech (Indic Parler-TTS)")
    st.caption("Notebook model: ai4bharat/indic-parler-tts. Generates local WAV audio for Hindi/Marathi.")

    mode = st.radio("Input mode", ["Direct Hindi/Marathi text", "Translate English then speak"], horizontal=True)
    language = st.selectbox("Language / voice language", ["Hindi", "Marathi"], key="tts_lang")
    speaker = st.selectbox("Speaker", SUPPORTED_SPEAKERS[language], key="tts_speaker")
    style = st.selectbox("Style", list(VOICE_STYLES.keys()), key="tts_style")

    text = ""
    if mode.startswith("Direct"):
        text = st.text_area("Text to speak", height=120, key="tts_direct_text")
    else:
        eng = st.text_area("English text", height=100, key="tts_eng_text")
        engine = st.selectbox("Translator", ["MarianMT", "Qwen2.5"], key="tts_engine")
        if eng and st.button("Translate for speech", key="tts_translate_btn"):
            if engine == "MarianMT":
                r = marian_translate(eng, language)
            else:
                r = translate_with_llm(eng, target_language=language)
            if r.get("success"):
                st.session_state["tts_translated"] = r.get("translated") or r.get("translated_text") or ""
            else:
                st.error(r.get("error"))
        text = st.text_area("Translated text to speak", value=st.session_state.get("tts_translated", ""), height=120, key="tts_translated_text")

    if st.button("Generate speech", type="primary", disabled=not (text and text.strip()), key="tts_generate_btn"):
        status = st.status("Loading model, preparing input, and generating audio...")
        with status:
            res = generate_speech(text, language=language, speaker=speaker, style=style)
        if res.get("success"):
            st.success(f"Generated {res.get('duration_seconds')}s audio on {res.get('device')}.")
            audio_path = res["audio_path"]
            st.audio(audio_path)
            with open(audio_path, "rb") as f:
                st.download_button("Download WAV", f, file_name=Path(audio_path).name, mime="audio/wav", key="tts_audio_download")
            st.caption(f"File: {audio_path}")
            st.json({k: v for k, v in res.items() if k != "audio_path"})
        else:
            st.error(res.get("error", "TTS failed."))
            st.info("If model access is gated, authenticate on the Hugging Face page and verify the model is available.")
