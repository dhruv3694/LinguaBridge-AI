# U1–U5 Requirement Matrix (Notebook-First Reconstruction)

Status legend: **COMPLETE** (implemented+tested where feasible), **PARTIALLY COMPLETE**, **BLOCKED** (needs model download/auth/hardware/time), **NOT IMPLEMENTED**, **FAILED**.

| Requirement | Unit | Notebook ref | Module | Streamlit page | Test | Status | Evidence/notes |
|---|---|---|---|---|---|---|---|
| MarianMT EN→HI | U2/U4 | cells 8,10,45 | `modules/translator.py` | Translator, Comparison, Bias | CUDA inference tested | **COMPLETE** | Verified with cached model on RTX 4050 |
| MarianMT EN→MR | U2/U4 | cells 8,10,45 | `modules/translator.py` | Translator, Comparison, Bias | CUDA inference tested | **COMPLETE** | Verified with cached model on RTX 4050 |
| Prompt zero/few/role | U1 | 16,18,22 | `modules/prompt_engineering.py` | Prompt Studio, Translator (Qwen) | unit tests | **COMPLETE** | Notebook logic preserved |
| Tones/registers | U1 | 16,20 | `modules/prompt_engineering.py` | Prompt Studio, Translator | unit tests | **COMPLETE** | 6 styles exposed |
| Qwen translation HI/MR | U1/U2/U4 | 26,30,32,35,45 | `modules/llm_translator.py` | Translator, Prompt Studio, Comparison, Bias | CUDA inference tested | **COMPLETE** | Uses `Qwen/Qwen2.5-1.5B-Instruct`; Hindi and Marathi verified |
| 20-sentence benchmark eval | U2/U5 | 41,43,47,49,51,53 | `modules/evaluation.py`, `data/translation_benchmark.csv` | Comparison | unit on metrics logic possible; full run blocked | **PARTIALLY COMPLETE** | UI runs subset; BLEU/chrF only with valid non-empty references (dataset has references) |
| BLEU/chrF (SacreBLEU) | U5 | 43,51 | `modules/evaluation.py` | Comparison | unit | **COMPLETE** (logic) | No fabricated metrics |
| Latency comparison | U2/U5 | 43,47,49 | `modules/evaluation.py` | Comparison | blocked for real | **PARTIALLY COMPLETE** | Implemented; real values require inference |
| BLIP captioning | U3/U4 | 59,67 | `modules/image_caption.py` | Image Captioning | CUDA inference tested | **COMPLETE** | `Salesforce/blip-image-captioning-base`; PIL and uploaded-image inputs supported |
| Image→caption→translation | U3/U4 | 69,71,73 | `modules/image_translation.py` | Image Captioning | CUDA end-to-end inference tested | **COMPLETE** | BLIP caption translated with MarianMT |
| Indic Parler-TTS HI/MR | U4 | 90,96,98,102,129 | `modules/speech.py` | TTS | CUDA inference tested | **COMPLETE** | `ai4bharat/indic-parler-tts`; requires parler-tts pkg & model |
| Speakers/styles (notebook) | U4 | 90 | `modules/speech.py` | TTS | unit | **COMPLETE** | HI: Divya/Rohit/Aman/Rani; MR: Sunita/Sanjay/Nikhil/Radha/Varun/Isha; Normal/Slow/Expressive |
| Gender bias dataset (20) | U5 | 134,135 | `data/bias_test_dataset.csv`, `modules/bias_evaluation.py` | Bias Evaluation | unit | **COMPLETE** | Neutral/masculine/feminine pairs |
| Gender marker detection HI/MR | U5 | 137,141 | `modules/bias_evaluation.py` | Bias Evaluation | unit | **COMPLETE** | Heuristic; human review required |
| Bias MarianMT+Qwen HI/MR | U5 | 145,147,149,151,153 | `modules/bias_evaluation.py` + translators | Bias Evaluation | blocked for real | **PARTIALLY COMPLETE** | UI runs selected subset/models |
| Bias exports/worksheet/chart | U5 | 149–157 | `modules/bias_evaluation.py` | Bias Evaluation | unit on data transforms | **COMPLETE** (logic) | CSV/PNG export functions present |
| HF session-only auth | Security | 86,119 (adapted securely) | `modules/hf_auth.py`, `services/authentication_service.py` | HF Authentication | unit on session helpers (mock st) | **COMPLETE** | No plaintext disk persistence; notebook token never reused |
| U1–U5 docs/matrix | Assignment | — | `docs/*` | U1–U5 Documentation, About | manual review | **PARTIALLY COMPLETE** | Docs created; real-inference evidence pending |

## Gaps / honesty
- GAN/VAE, fine-tuning, BERT-family, GPT-style generator, video/music generation: **not present as implemented notebook model workflows** in the extracted allowlist/primary cells reviewed; no additional HF models added. Assignment-specific gaps must be confirmed against your course rubric.
- Runtime checks confirmed CUDA inference for MarianMT, Qwen, BLIP and Parler-TTS in the validated Python 3.11 environment.
