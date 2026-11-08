# Model Card (Summary)

## Intended use
Academic English→Hindi/Marathi translation, caption translation, speech synthesis, and responsible-AI evaluation (P19).

## Models (notebook allowlist only)
1. `Helsinki-NLP/opus-mt-en-hi` — EN→HI MarianMT.
2. `Helsinki-NLP/opus-mt-en-mr` — EN→MR MarianMT.
3. `Qwen/Qwen2.5-1.5B-Instruct` — prompt-based LLM translation.
4. `Salesforce/blip-image-captioning-base` — image captioning.
5. `ai4bharat/indic-parler-tts` — Hindi/Marathi TTS.

## Limitations
- Translation errors, hallucinations, register mismatch, and gender bias possible.
- Bias module uses heuristic markers and is not a full linguistic bias audit.
- Automatic metrics require valid human references.
- IndicTrans2 notebook cells are not included in the running application.

## Ethical considerations
Do not use outputs for high-stakes decisions without human review. Protect user tokens and personal data.
