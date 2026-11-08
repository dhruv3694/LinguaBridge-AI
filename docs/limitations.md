# Limitations
- **References missing/quality:** automatic BLEU/chrF can be invalid if human references are absent or low quality.
- **Heuristic bias detection:** selected marker lists do not cover full Hindi/Marathi morphology or pragmatics; false positives/negatives possible.
- **Model behavior:** LLMs may hallucinate or ignore constraints; MarianMT may produce literal output.
- **Hardware:** 3B/1.5B/TTS models may be slow or OOM on 6 GB VRAM; CPU fallback can be very slow.
- **Access:** some HF models may require token acceptance of terms; network required for first download.
- **Security:** never trust `trust_remote_code` models without review; keep tokens session-only.
- **Notebook parity:** Colab cells used `/content`, `input()`, `display()`; app adapts behavior to Streamlit/local paths.
