# ✍️ LlamaWrites v2.0 — AI Blog Generator

> Generate structured, professional blog posts and matching AI illustrations — 100% locally, no API key required.

---

## What is LlamaWrites?

LlamaWrites is a Streamlit-based web application that uses locally-running open-source AI models to:

1. **Generate a full blog post** in your chosen style and tone.
2. **Generate a matching illustration** using an LCM-accelerated image model (4× faster than standard SD).
3. **Display everything in a clean, professional UI** with Light and Dark themes.
4. **Let you download** the blog as `.md`, `.txt`, or `.docx`, and the image as `.png`.
5. **Regenerate** blog or image independently without re-entering your settings.

Everything runs on your own machine. No OpenAI key, no Hugging Face token, no paid API.

---

## Features

- **📝 Blog Only mode** — text model only; image model is never loaded
- **📝 + 🖼️ Blog + Image mode** — image model loads lazily, only when needed
- **10 Blog Styles** — Informative, Research, Storytelling, How-To, Opinion, Listicle, News, SEO, Conversational, Professional
- **7 Tones** — Professional, Friendly, Conversational, Academic, Persuasive, Inspirational, Neutral
- **🔄 Regenerate Blog** — get a new variation without re-entering settings
- **🔄 Regenerate Image** — regenerate only the image, blog is preserved
- **⬇️ Download** — `.md`, `.txt`, `.docx` for blog; `.png` for image
- **☀️ / 🌙 Light / Dark theme** — toggle in the sidebar
- **⚡ LCM image model** — 4 diffusion steps instead of 20 (~5× faster on CPU)
- **🔁 Streamlit caching** — models load once per session, never reloaded on rerun
- **⚠️ Graceful error handling** — image failure never destroys the generated blog
- **📊 Word count feedback** — requested vs. actual words displayed

---

## Tech Stack

| Component        | Technology                                        |
|------------------|---------------------------------------------------|
| UI Framework     | Streamlit 1.30+                                   |
| Text Model       | Qwen/Qwen2.5-0.5B-Instruct (fast, CPU-friendly)  |
| Image Model      | SimianLuo/LCM_Dreamshaper_v7 (LCM, 4-step)       |
| Image Fallback   | nota-ai/bk-sdm-small                              |
| ML Framework     | PyTorch + Hugging Face Transformers               |
| Diffusion        | Hugging Face Diffusers (with LCMScheduler)        |
| Image processing | Pillow                                            |
| Document export  | python-docx                                       |
| Language         | Python 3.12                                       |

---

## Blog Styles

| Style | Icon | Best For |
|-------|------|---------|
| Informative / Knowledge | 📚 | Educational content, explainers |
| Research / Analytical | 🔬 | Evidence-based, structured writing |
| Storytelling | 📖 | Narrative posts, personal stories |
| How-To / Tutorial | 🛠️ | Step-by-step guides |
| Opinion / Thought Leadership | 💡 | Professional opinion pieces |
| Listicle | 📋 | Numbered lists, quick-read posts |
| News / Current Affairs | 📰 | Journalistic, neutral reporting |
| SEO Blog | 🔍 | Search-optimised web content |
| Conversational | 💬 | Friendly, casual writing |
| Professional | 💼 | Formal business content |

Each style directly affects the **prompt structure** sent to the model, producing meaningfully different outputs.

---

## Hardware Requirements

| Mode     | Minimum RAM | Recommended RAM | GPU               |
|----------|-------------|-----------------|-------------------|
| CPU only | 8 GB        | 16 GB           | Not required      |
| GPU      | 6 GB VRAM   | 8 GB VRAM       | NVIDIA (CUDA 11+) |

---

## Software Requirements

- Windows 10 or 11
- Python 3.11 or 3.12
- Git (to clone)
- Internet connection (first run only, for model downloads)

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/techy-ops/LlamaWrites.git
cd LlamaWrites
```

### 2. Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch

**CPU-only (most users):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.1):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the application

```powershell
streamlit run "LlamaWrite Code.py"
```

The app opens automatically at [http://localhost:8501](http://localhost:8501).

---

## First-Run: Model Downloads

On the very first run, models are downloaded automatically and cached:

| Model                          | Size    | Purpose                    | When downloaded      |
|--------------------------------|---------|----------------------------|----------------------|
| EleutherAI/gpt-neo-1.3B        | ~2.6 GB | Blog text generation       | Always (first run)   |
| SimianLuo/LCM_Dreamshaper_v7   | ~2.0 GB | Image generation (LCM)     | Blog + Image mode only |

**Blog Only mode total first-run download: ~2.6 GB**  
**Blog + Image mode total first-run download: ~4.6 GB**

Models are saved to the Hugging Face cache (`C:\Users\<you>\.cache\huggingface\`). Subsequent runs use local cache and do **not** re-download.

---

## Usage

1. Open the app at [http://localhost:8501](http://localhost:8501).
2. **Choose a mode** in the sidebar: Blog Only or Blog + Image.
3. Enter a **blog topic**.
4. Select a **Blog Style** (e.g. Storytelling, Listicle, How-To).
5. Select a **Tone** (e.g. Conversational, Academic, Professional).
6. Set the desired **word count** (100–1,500).
7. Click **Generate**.
8. Read and download your blog. If Blog + Image mode, wait for the image.
9. Use **Regenerate Blog** for a new variation, **Regenerate Image** for a new image.

---

## Performance

### CPU vs GPU

| Operation              | CPU (typical)   | GPU NVIDIA RTX 3070 |
|------------------------|-----------------|---------------------|
| Load text model        | 30–60 s         | 10–20 s             |
| Generate 300 words     | 30–60 s         | 5–10 s              |
| Load image model       | 60–120 s        | 15–30 s             |
| Generate image (4 LCM) | **2–5 min**     | **5–15 s**          |

> 💡 The image model uses Latent Consistency Model (LCM) distillation: **4 inference steps** instead of 20. This provides approximately a **5× speedup** on CPU compared to the previous standard SD model.

---

## Troubleshooting

### App won't start
- Activate the venv: `venv\Scripts\activate`
- Verify Streamlit: `pip show streamlit`

### Model download fails
- Check your internet connection
- Try again — Hugging Face sometimes has transient errors
- If behind a corporate proxy, configure proxy settings

### Out of memory during text generation
- Lower the word count (try 100–200)
- Close other memory-intensive applications

### Out of memory during image generation
- Switch to Blog Only mode
- Reduce other running applications
- Image generation requires at least 3–4 GB of RAM on CPU

### DOCX download button not showing
- Run: `pip install python-docx`

### Image model takes too long
- The LCM model uses 4 steps (not 20). On CPU, 2–5 minutes is expected.
- For zero wait on images, use Blog Only mode.

---

## Known Limitations

- **GPT-Neo-1.3B is not instruction-tuned.** It generates text by continuation. Quality is substantially improved by the style/tone prompt system, but it is not comparable to GPT-4 or Gemini.
- **Word count is approximate.** Output may vary ±15–25% from requested count due to tokenisation differences.
- **Maximum output is ~1,400 words** due to the model's 2,048-token context window.
- **Image generation on CPU is still slow.** LCM reduces the time significantly (2–5 min vs 10–20 min), but generating images on CPU will never be instant.
- **Image quality** is limited by the LCM distilled model. It produces decent blog-header-style images but not photorealistic output.

---

## File Structure

```
LlamaWrites/
├── LlamaWrite Code.py   # Main Streamlit application (v2.0)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── offload/             # Temporary model offload directory (auto-created)
└── venv/                # Virtual environment (created by you)
```

---

## Changelog

### v2.0
- Added **Blog Only / Blog + Image** generation modes
- Image model is now **lazy-loaded** — never loaded in Blog Only mode
- Switched image model to **LCM_Dreamshaper_v7** (4 steps, ~5× faster)
- Added **10 blog styles** with style-specific prompts
- Added **7 tone options** that affect generated content
- Added **Regenerate Blog** button (blog only, image preserved)
- Added **Regenerate Image** button (image only, blog preserved)
- Added **DOCX download** option
- Added **Light / Dark theme** toggle
- Blog preserved in `session_state` — survives image failure
- Fixed `OFFLOAD_DIR` path bug (was crashing on non-project CWD)
- Professional UI redesign with Inter font and card-based layout
- Comprehensive error handling — no raw Python errors shown to users

---

## License

Open source. Original repository: [github.com/techy-ops/LlamaWrites](https://github.com/techy-ops/LlamaWrites)