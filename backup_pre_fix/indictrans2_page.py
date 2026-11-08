"""IndicTrans2 experimental workflow (notebook cells 114-129). Separate from MarianMT/Qwen."""
from __future__ import annotations
import streamlit as st
from modules.translation import MODEL_NAME, translate_text as indictrans2_translate
from modules.speech import SUPPORTED_SPEAKERS, VOICE_STYLES, generate_speech


def render():
    st.title("IndicTrans2 Translation (Experimental)")
    st.warning(
        f"This page uses the notebook's later IndicTrans2 experiment (`{MODEL_NAME}`) with IndicTransToolkit. "
        "It is separate from the primary MarianMT/Qwen translation engines and comparison."
    )
    st.caption("Requires `IndicTransToolkit` and may require Hugging Face access/token depending on the model repository.")
    target = st.selectbox("Target language", ["Hindi", "Marathi"], key="it2_lang")
    text = st.text_area("English text", "Education is the most powerful tool for creating a better future.", key="it2_text")
    if st.button("Translate with IndicTrans2", type="primary"):
        with st.spinner("Loading IndicTrans2 and translating..."):
            res = indictrans2_translate(text, target)
        if res.get("success"):
            st.success(res.get("translated"))
            st.session_state["it2_last"] = res.get("translated")
            st.json({k: v for k, v in res.items() if k != "translated"})
        else:
            st.error(res.get("error"))
            st.info("If the model is gated, authenticate on the Hugging Face Authentication page.")
    st.divider()
    st.subheader("Translation + Speech workflow (notebook)")
    speech_text = st.text_area("Text to speak (Hindi/Marathi)", value=st.session_state.get("it2_last", ""), key="it2_speech")
    speaker = st.selectbox("Speaker", SUPPORTED_SPEAKERS.get(target, ["Divya"]), key="it2_spk")
    style = st.selectbox("Style", list(VOICE_STYLES.keys()), key="it2_style")
    if st.button("Generate speech from above text"):
        with st.spinner("Generating speech..."):
            audio = generate_speech(speech_text, language=target, speaker=speaker, style=style)
        if audio.get("success"):
            st.audio(audio["audio_path"])
            with open(audio["audio_path"], "rb") as f:
                st.download_button("Download WAV", f, file_name="indictrans2_speech.wav", mime="audio/wav")
        else:
            st.error(audio.get("error"))
