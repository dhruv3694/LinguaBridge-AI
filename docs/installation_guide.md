# Installation Guide (Windows, PowerShell) — Tested on this machine

## Machine used for validation
- OS: Windows 11, AMD64
- CPU: AMD Ryzen 7 7735HS
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, driver 596.36, 6 GB VRAM
- Python at validation time: 3.14.5 (global), 3.11.0 (venv) — **the project runs on Python 3.11**
- C: drive was nearly full (~0.2–1.4 GB free); all environments/caches live on **D:**

## 1. Use one interpreter for everything
Run pip and Streamlit through the **same** Python executable:
```powershell
python -m pip ...
python -m streamlit run app.py
```
Do not mix a global `streamlit.exe` from another Python with the project interpreter.

## 2. Create and activate the project virtual environment
```powershell
# From any location; the venv lives on D: (C: was too small for PyTorch wheels)
py -3.11 -m venv D:\LinguaBridge_Environment\venv311
D:\LinguaBridge_Environment\venv311\Scripts\Activate.ps1
```
Verify:
```powershell
python --version          # Python 3.11.x
python -c "import sys; print(sys.executable)"   # ...venv311\Scripts\python.exe
python -m pip --version
```

## 3. Cache/temp locations on D: (recommended while C: is small)
The app automatically configures model downloads and Python temporary files to use:
```text
D:\LinguaBridge_Environment\hf-cache
D:\LinguaBridge_Environment\hf-cache\tmp
```
You can override the cache root with `LINGUABRIDGE_CACHE_DIR`. For pip installs, set these in PowerShell:
```powershell
$env:PIP_CACHE_DIR = "D:\LinguaBridge_Environment\pip-cache"
```
The Streamlit app configures its Hugging Face and temporary-file paths at startup; only the pip download cache is set manually above.

## 4. Install PyTorch (CUDA) — required for model inference
Validated command:
```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```
Verified result on this machine:
```text
PyTorch: 2.11.0+cu128
CUDA available: True
CUDA version: 12.8
Device: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6.00 GB
```
CPU-only fallback: `python -m pip install torch` (official index).

## 5. Install project dependencies
```powershell
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-optional.txt   # optional: Parler-TTS
```
Validated core set (Python 3.11 venv):
```text
torch 2.11.0+cu128
transformers 4.46.1
huggingface_hub 0.36.2
tokenizers 0.20.3
safetensors 0.8.0
accelerate 1.15.0
sentencepiece 0.2.2
sacremoses 0.2.0
joblib 1.6.0
streamlit 1.67.0
pandas 3.1.1
numpy 2.4.6
Pillow 13.0.0
sacrebleu 2.6.0
matplotlib 3.11.2
protobuf 6.33.6
python-dotenv 1.0.0 (from user site)
pytest 9.2.0
```
Note: `sentencepiece 0.2.3` ships generated Protobuf code requiring Protobuf 7.36.1 and fails with the Protobuf 6 runtime. The project constrains SentencePiece below 0.2.3 and keeps Protobuf 6 for Streamlit compatibility. Real Parler-TTS Hindi and Marathi WAV generation was verified with the cached model.
The installed `descript-audiotools 0.7.4` metadata requests Protobuf `<5`, conflicting with Streamlit's Protobuf `>=5.26.1` requirement. Do not downgrade Protobuf to satisfy that optional audio package's metadata: Streamlit import and real Parler-TTS inference were verified with Protobuf 6.33.6.

## 6. Diagnose
```powershell
python scripts/check_environment.py
python -m pip check
```

## 7. Run the application (only with the venv activated)
Recommended:
```powershell
D:\LinguaBridge_Environment\venv311\Scripts\Activate.ps1
# the app configures HF_HOME, HF_HUB_CACHE, and TEMP under D:
python -m streamlit run app.py
```
Or use the startup script which verifies the interpreter and PyTorch first:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_app.ps1
```

## 8. Hugging Face authentication
Use the **Hugging Face Authentication** page in the app (masked input). Token is stored in Streamlit session state only — never in `.env`, source, CSV/JSON, or logs.
- Public models (MarianMT, BLIP) may download without a token.
- Gated models (e.g. `ai4bharat/indic-parler-tts`) require: token entered in the app **and** accepting the model's terms on its Hugging Face model card. The user must do this themselves — no token is stored by this project.
- The original notebook contained a plaintext token; treat it as compromised and revoke it.

## Hardware notes
- RTX 4050 6 GB VRAM: models load lazily, one at a time. Qwen 1.5B direct CUDA load can exhaust the Windows paging file when C: is nearly full — free space on C: or move the Windows page file to D: if Qwen fails with "paging file is too small".
- Do not load all heavy models simultaneously.

## Test
```powershell
python -m pytest -q
```
