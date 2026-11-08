# Model Inventory (Application Allowlist)
The notebook is retained as a development reference. **The running app uses only these five Hugging Face model IDs.**

| # | Model | Exact HF ID | Notebook refs | Purpose | Loading (notebook intent) | Auth |
|---|---|---|---|---|---|---|
| 1 | MarianMT EN→HI | `Helsinki-NLP/opus-mt-en-hi` | 8,10,45,69,145 | English→Hindi seq2seq translation | `MarianTokenizer`/`MarianMTModel`, beam search (`num_beams=4`), lazy cache | Usually public; token passed if needed |
| 2 | MarianMT EN→MR | `Helsinki-NLP/opus-mt-en-mr` | 8,10,45,69,145 | English→Marathi seq2seq translation | Same as above | Same |
| 3 | Qwen Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | 26,30,32,35,45,145 | Prompt-based LLM translation (HI/MR) | `AutoTokenizer`/`AutoModelForCausalLM`, chat template, greedy `do_sample=False`, `max_new_tokens=160`, `repetition_penalty=1.05` | May require token/terms depending on access |
| 4 | BLIP base | `Salesforce/blip-image-captioning-base` | 59,67,69,71 | English image captioning | `BlipProcessor`/`BlipForConditionalGeneration`, beams=4 | Usually public |
| 5 | Indic Parler-TTS | `ai4bharat/indic-parler-tts` | 81,90,96,98,102,108,129 | Hindi/Marathi speech (WAV) | `ParlerTTSForConditionalGeneration` + text/description tokenizers; speaker+style descriptions | May require HF token/access |

## Notes
- **TTS decision:** Vyom/OpenXFI models removed from implementation; notebook uses `ai4bharat/indic-parler-tts` only.
- The archived notebook's optional IndicTrans2 experiment is not part of the application; the app uses MarianMT/Qwen for translation and Indic Parler-TTS for speech.
- Review each model card/license on Hugging Face before redistribution/deployment.
- This app does not introduce additional HF models.
