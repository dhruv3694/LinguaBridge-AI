# Public GPU deployment (Hugging Face Spaces)

This app can be deployed as a public Docker Space. The interface and public
models are available to visitors; gated-model access is provided by each
visitor's own Hugging Face token entered in the app. Do not configure a shared
`HF_TOKEN` Space secret for this setup.

## 1. Push the latest project to GitHub

The project Git repository is prepared locally. Push the `main` branch after
signing into GitHub from VS Code or Git Credential Manager:

```powershell
git push -u origin main
```

## 2. Create the Space

1. Sign in at [Hugging Face](https://huggingface.co/).
2. Select **New Space** and name it, for example, `linguabridge-ai`.
3. Choose **Docker** as the SDK and **Public** visibility.
4. Create the Space, then upload or sync the latest GitHub project files,
   including the root `README.md` (with the Space metadata), `.dockerignore`,
   `Dockerfile`, and `requirements-deploy.txt`.
5. In Space **Settings → Hardware**, select a GPU tier with enough VRAM for
   one model at a time. CUDA inference has been verified locally on a 6 GB
   RTX 4050; a larger tier gives more headroom for concurrent visitors.
6. If available for your Space plan, attach persistent storage for `/data`.
   The Docker image stores Hugging Face downloads and temporary files under
   `/data`; without persistent storage, model files may need to download again
   after a rebuild or restart.

The Space builds the Docker image and displays its public URL when the app is
ready. The Docker Space exposes Streamlit on port 7860.

## 3. Hugging Face model access

- Visitors enter their own Hugging Face token in the app when a model requires
  authentication. Tokens are held in that visitor's Streamlit session and are
  not written to the repository.
- Each visitor must independently accept any applicable model terms.
- Never commit tokens to GitHub or put a personal token in a public Space
  secret. Public GPU hardware can incur charges and be used by anyone.
- Consider pausing the Space when it is not in use and review the host's
  hardware, storage, and usage settings.

## Local checks before publishing

```powershell
python -m pytest -q
docker build -t linguabridge-ai .
docker run --rm -p 7860:7860 linguabridge-ai
```

GPU inference requires a GPU-enabled Space runtime. A local Docker build alone
does not provide GPU access.
