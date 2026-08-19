"""
LlamaWrites — AI Blog Generator  v2.0
Streamlit app that generates structured blog posts and optional matching images
using local AI models. Runs 100% offline — no API key required.
"""

import os
import streamlit as st
import torch
import re
import io
from pathlib import Path

# Suppress verbose HuggingFace advisory warnings
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

# ──────────────────────────────────────────────────────────────────────────────
# Page config — MUST be the very first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LlamaWrites — AI Blog Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state — initialise defaults once
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "theme":           "dark",
    "gen_mode":        "Blog Only",
    "blog_text":       None,
    "blog_image":      None,
    "last_topic":      "",
    "last_style":      "",
    "last_tone":       "",
    "last_wc":         300,
    "blog_generated":  False,
    "image_generated": False,
    "image_error":     None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
TEXT_MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
IMAGE_MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
IMAGE_FALLBACK = "nota-ai/bk-sdm-small"

OFFLOAD_DIR = Path(__file__).parent / "offload"
OFFLOAD_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    DEVICE_LABEL = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
else:
    DEVICE_LABEL = "CPU"

# ──────────────────────────────────────────────────────────────────────────────
# Blog styles
# ──────────────────────────────────────────────────────────────────────────────
BLOG_STYLES = {
    "Informative / Knowledge": {
        "icon": "📚",
        "desc": "Educational, factual, beginner-friendly",
        "prompt_intro": (
            "Write a clear, educational, and informative blog post about '{topic}'. "
            "Explain key concepts, provide factual information, and make it accessible "
            "to readers with no prior knowledge. Use clear headings and helpful examples."
        ),
        "structure": "Introduction - Key Concepts - Detailed Explanation - Practical Takeaways - Conclusion",
    },
    "Research / Analytical": {
        "icon": "🔬",
        "desc": "Evidence-based, structured, professional",
        "prompt_intro": (
            "Write a research-oriented, analytical blog post about '{topic}'. "
            "Present evidence, discuss multiple perspectives, analyze trends, "
            "and draw well-reasoned conclusions with a structured, professional format."
        ),
        "structure": "Abstract - Background - Analysis - Evidence - Discussion - Conclusion",
    },
    "Storytelling": {
        "icon": "📖",
        "desc": "Narrative-driven, engaging, emotional",
        "prompt_intro": (
            "Write an engaging, narrative-driven blog post about '{topic}'. "
            "Open with a compelling story or anecdote, weave vivid descriptions "
            "throughout, and use storytelling to emotionally connect with the reader."
        ),
        "structure": "Opening Hook (story) - Rising Action - Core Message - Personal Reflection - Inspiring Close",
    },
    "How-To / Tutorial": {
        "icon": "🛠️",
        "desc": "Step-by-step, practical, action-oriented",
        "prompt_intro": (
            "Write a practical, step-by-step tutorial blog post about '{topic}'. "
            "Include numbered steps, clear instructions, and useful tips for beginners. "
            "State what readers will achieve by the end."
        ),
        "structure": "What You Will Learn - Prerequisites - Step 1 - Step 2 - More Steps - Tips - Summary",
    },
    "Opinion / Thought Leadership": {
        "icon": "💡",
        "desc": "Strong perspective, arguments, professional",
        "prompt_intro": (
            "Write a thought-leadership opinion piece about '{topic}'. "
            "Take a clear, well-argued position, back it with reasoning and evidence, "
            "anticipate counterarguments, and end with a powerful conclusion."
        ),
        "structure": "Bold Opening Statement - Core Argument - Supporting Evidence - Counterargument - Strong Conclusion",
    },
    "Listicle": {
        "icon": "📋",
        "desc": "Numbered sections, scannable, social-friendly",
        "prompt_intro": (
            "Write an engaging listicle blog post about '{topic}'. "
            "Structure as a numbered list of key points or tips. "
            "Each item should have a short heading and 2-3 sentences of explanation."
        ),
        "structure": "Introduction - Item 1 - Item 2 - Item 3 - More Items - Wrap-Up",
    },
    "News / Current Affairs": {
        "icon": "📰",
        "desc": "Journalistic, concise, neutral tone",
        "prompt_intro": (
            "Write a news-style blog post about '{topic}'. "
            "Use an inverted pyramid: lead with the most important information, "
            "follow with context and details, end with background and implications."
        ),
        "structure": "Lead (who/what/when/where/why) - Key Details - Context - Implications - Closing",
    },
    "SEO Blog": {
        "icon": "🔍",
        "desc": "Search-optimised, structured, readable",
        "prompt_intro": (
            "Write an SEO-optimised blog post about '{topic}'. "
            "Use the topic naturally throughout, include clear headings, "
            "write in short readable paragraphs, and provide genuine value."
        ),
        "structure": "Introduction with keyword - What is it - Why it matters - How to use it - Conclusion",
    },
    "Conversational": {
        "icon": "💬",
        "desc": "Friendly, natural, easy-to-read",
        "prompt_intro": (
            "Write a warm, conversational blog post about '{topic}'. "
            "Talk to the reader directly as if having a friendly discussion. "
            "Use a relaxed tone, contractions, and relatable language."
        ),
        "structure": "Casual Open - Chat-style Exploration - Personal Take - Reader Takeaway - Friendly Close",
    },
    "Professional": {
        "icon": "💼",
        "desc": "Formal, business-oriented, polished",
        "prompt_intro": (
            "Write a polished, professional blog post about '{topic}'. "
            "The writing should be formal, precise, and suitable for business audiences. "
            "Use well-structured paragraphs and authoritative statements."
        ),
        "structure": "Executive Summary - Context - Core Analysis - Key Insights - Recommendations - Conclusion",
    },
}

TONES = {
    "Professional":   "Write in a professional, clear, and precise tone.",
    "Friendly":       "Write in a warm, friendly, and approachable tone.",
    "Conversational": "Write in a natural, conversational tone as if talking to a friend.",
    "Academic":       "Write in a formal academic tone with structured arguments and precise language.",
    "Persuasive":     "Write in a compelling, persuasive tone that motivates the reader to think or act.",
    "Inspirational":  "Write in an uplifting, motivational tone that energises and inspires the reader.",
    "Neutral":        "Write in a neutral, balanced, objective tone without strong opinion.",
}

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Light and Dark themes
# ──────────────────────────────────────────────────────────────────────────────
def _get_css(theme: str) -> str:
    if theme == "dark":
        bg        = "#0f1117"
        surface   = "#1e293b"
        surface2  = "#0f172a"
        border    = "#334155"
        text      = "#e2e8f0"
        subtext   = "#94a3b8"
        accent    = "#6366f1"
        accent_lt = "#818cf8"
        warning   = "#f59e0b"
        badge_bg  = "#1e3a5f"
        badge_txt = "#93c5fd"
        input_bg  = "#1e293b"
        hdr_grad  = "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
        dev_bg    = "#064e3b"
        dev_txt   = "#6ee7b7"
        cpu_bg    = "#1c1917"
        cpu_txt   = "#fde68a"
    else:
        bg        = "#f8fafc"
        surface   = "#ffffff"
        surface2  = "#f1f5f9"
        border    = "#e2e8f0"
        text      = "#1e293b"
        subtext   = "#64748b"
        accent    = "#4f46e5"
        accent_lt = "#6366f1"
        warning   = "#d97706"
        badge_bg  = "#ede9fe"
        badge_txt = "#4c1d95"
        input_bg  = "#ffffff"
        hdr_grad  = "linear-gradient(135deg, #312e81 0%, #4338ca 50%, #4f46e5 100%)"
        dev_bg    = "#f0fdf4"
        dev_txt   = "#166534"
        cpu_bg    = "#fffbeb"
        cpu_txt   = "#92400e"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {bg} !important;
    color: {text} !important;
}}
.stApp {{
    background-color: {bg} !important;
}}
.lw-header {{
    background: {hdr_grad};
    padding: 2.5rem 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.2);
}}
.lw-header h1 {{
    color: #f8fafc;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}}
.lw-header p {{
    color: #c7d2fe;
    font-size: 1.05rem;
    margin: 0;
}}
.blog-content {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-top: 1rem;
    line-height: 1.85;
    color: {text};
    font-size: 1rem;
}}
.blog-content h1, .blog-content h2, .blog-content h3 {{
    color: {accent_lt};
    margin: 1.5rem 0 0.6rem;
}}
.section-header {{
    font-size: 0.9rem;
    font-weight: 700;
    color: {subtext};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.8rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid {border};
}}
.badge {{
    background: {badge_bg};
    color: {badge_txt};
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-block;
    margin: 0 0.4rem 0.4rem 0;
}}
.badge-device {{
    background: {dev_bg};
    color: {dev_txt};
    border-radius: 8px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-block;
    margin-right: 0.5rem;
}}
.cpu-notice {{
    background: {cpu_bg};
    border-left: 4px solid {warning};
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.2rem;
    color: {cpu_txt};
    font-size: 0.88rem;
    margin: 0.75rem 0;
}}
.stButton>button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}}
.stButton>button[kind="primary"] {{
    background: {accent} !important;
    border: none !important;
    color: white !important;
}}
.stButton>button[kind="primary"]:hover {{
    background: {accent_lt} !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}}
.stTextInput>div>div>input,
.stNumberInput>div>div>input {{
    background: {input_bg} !important;
    color: {text} !important;
    border-radius: 10px !important;
}}
[data-testid="stSidebar"] {{
    background: {surface2} !important;
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
</style>
"""


st.markdown(_get_css(st.session_state.theme), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model loaders — @st.cache_resource = load once per session
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_text_model():
    """
    Load Qwen/Qwen2.5-0.5B-Instruct.
    Uses @st.cache_resource so it is initialised exactly once per session.
    Returns (model, tokenizer) tuple or None on failure.
    """
    import os
    if DEVICE == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    ph = st.empty()
    try:
        ph.info(f"Loading text model (`{TEXT_MODEL_ID}`)...")
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(DEVICE)
        model.eval()
        ph.success(f"Text model ready on {DEVICE_LABEL}")
        return model, tokenizer
    except MemoryError:
        ph.error("Not enough memory to load the text model. Close other applications and retry.")
        return None
    except OSError as e:
        msg = str(e)
        if "404" in msg or "not found" in msg.lower():
            ph.error("Text model not found. Check your internet connection for first-run download.")
        else:
            ph.error(f"Could not load text model: {msg}")
        return None
    except Exception as e:
        ph.error(f"Unexpected error loading text model. Restart the app and try again. Details: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_image_model():
    """
    Load SimianLuo/LCM_Dreamshaper_v7 with LCMScheduler.
    Uses only 4 diffusion steps — approximately 5x faster than standard SD on CPU.
    Falls back to nota-ai/bk-sdm-small if LCM fails.

    IMPORTANT: This function is NEVER called for Blog Only mode.
    It is only called when the user explicitly selects Blog + Image.
    """
    from diffusers import DiffusionPipeline, StableDiffusionPipeline, LCMScheduler

    ph = st.empty()

    # Try LCM first
    try:
        ph.info(
            f"Loading image model (`{IMAGE_MODEL_ID}`)... "
            "Using LCM (4 steps) for fast CPU generation."
        )
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        pipe = DiffusionPipeline.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=dtype,   # diffusers still uses torch_dtype
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        ph.success(f"Image model ready on {DEVICE_LABEL} (LCM, 4-step mode)")
        return pipe, "lcm"
    except Exception as lcm_err:
        ph.warning(f"LCM model failed ({lcm_err}). Trying fallback...")

    # Fallback
    try:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            IMAGE_FALLBACK,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        ph.success(f"Image model ready on {DEVICE_LABEL} (fallback: bk-sdm-small)")
        return pipe, "sdm"
    except Exception as fb_err:
        ph.error(
            f"All image models failed to load. Reason: {fb_err}. "
            "Your blog is still available — image generation is disabled."
        )
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

def build_blog_prompt(topic: str, word_count: int, style: str, tone: str) -> list[dict]:
    style_cfg = BLOG_STYLES.get(style, BLOG_STYLES["Informative / Knowledge"])
    tone_instruction = TONES.get(tone, TONES["Professional"])
    style_intro = style_cfg["prompt_intro"].replace("{topic}", topic)
    structure = style_cfg["structure"]

    system_msg = (
        "You are an expert, professional blog writer. "
        "Your task is to write high-quality, engaging, and well-structured blog posts "
        "strictly following the requested style, tone, and approximate word count. "
        "Use Markdown formatting with headings (## and ###), clear paragraphs, and key takeaways where appropriate. "
        "Do not include meta-commentary, introductory notes, or disclaimers. Begin directly with the blog title or first heading."
    )

    user_msg = (
        f"Topic: {topic}\n"
        f"Blog Style: {style}\n"
        f"Style Instructions: {style_intro}\n"
        f"Desired Structure: {structure}\n"
        f"Tone: {tone_instruction}\n"
        f"Target Length: Approximately {word_count} words.\n\n"
        f"Write the complete blog post now."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_image_prompt(topic: str, style: str) -> str:
    """
    Build a focused, keyword-rich visual prompt that produces accurate,
    topic-relevant imagery without confusing filler phrases.
    """
    clean_topic = topic.strip().strip('"').strip("'")
    style_visual_tags = {
        "Informative / Knowledge":   "crisp professional photography, clear subject lighting, realistic, detailed, 4k",
        "Research / Analytical":     "high-tech concept visualization, clean composition, crisp focus, modern aesthetic",
        "Storytelling":              "cinematic wide-angle shot, atmospheric lighting, dramatic depth, rich color grading, 8k",
        "How-To / Tutorial":         "clear practical studio photography, bright natural daylight, clean background, sharp detail",
        "Opinion / Thought Leadership": "bold editorial magazine photography, striking subject contrast, artistic composition",
        "Listicle":                  "vibrant modern aesthetic, clean flat composition, sharp focus, colorful studio shot",
        "News / Current Affairs":    "authentic photojournalism, natural documentary lighting, real-world context, 4k",
        "SEO Blog":                  "modern digital banner photography, clean aesthetic, sharp focus, balanced lighting",
        "Conversational":            "warm inviting atmosphere, natural golden hour lighting, authentic, high quality",
        "Professional":              "corporate executive setting, sleek modern architecture, polished professional photography",
    }
    visual_tag = style_visual_tags.get(style, "detailed realistic photography, cinematic lighting, 8k")
    return f"{clean_topic}, {visual_tag}"


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def post_process_blog(raw_text: str, topic: str) -> str:
    text = raw_text.strip()

    # Collapse 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Ensure it has a title heading
    if not text.startswith("#"):
        text = f"# {topic}\n\n" + text

    return text.strip()


def estimate_max_new_tokens(word_count: int) -> int:
    # 1 word ~ 1.25 tokens. Budget precisely to prevent slow over-generation on CPU.
    return max(60, min(int(word_count * 1.25), 900))


# ──────────────────────────────────────────────────────────────────────────────
# Generation functions
# ──────────────────────────────────────────────────────────────────────────────

def generate_blog(topic: str, word_count: int, style: str, tone: str, text_resources) -> str | None:
    if text_resources is None:
        st.error("Text model failed to load. Cannot generate blog. Please restart the app.")
        return None

    model, tokenizer = text_resources
    messages = build_blog_prompt(topic, word_count, style, tone)
    max_new_tokens = estimate_max_new_tokens(word_count)

    # Stop generation cleanly on EOS or im_end tokens
    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int):
        eos_ids.append(im_end_id)

    try:
        with st.spinner(f"✍️ Writing your {style} blog about '{topic}'... (~{word_count} words)"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.85,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=eos_ids,
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return post_process_blog(raw, topic)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error("Out of memory during blog generation. Try reducing the word count.")
        else:
            st.error(f"A runtime error occurred during blog generation: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during blog generation. Please try again. Details: {e}")
        return None


def run_image_generation(topic: str, style: str, pipe, pipe_type: str):
    if pipe is None:
        st.warning("Image model is not available. Image generation skipped.")
        return None

    img_prompt = build_image_prompt(topic, style)
    neg_prompt = "blurry, low quality, distorted, deformed, text, watermark, bad anatomy, cartoon, drawing, sketch"

    try:
        if pipe_type == "lcm":
            # LCM models require low guidance (1.0–2.0). 1.5 gives high topic adherence with single forward pass speed.
            steps = 4
            guidance = 1.5
        else:
            steps = 8 if DEVICE == "cpu" else 20
            guidance = 5.0

        # 384x384 reduces CPU diffusion compute by 44% compared to 512x512, cutting generation time significantly.
        img_size = 384 if DEVICE == "cpu" else 512
        est = "1–2 min" if DEVICE == "cpu" else "~10 sec"

        with st.spinner(f"🎨 Generating image for '{topic}' ({steps} steps, {DEVICE.upper()})... Estimated: {est}"):
            with torch.inference_mode():
                output = pipe(
                    img_prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=img_size,
                    width=img_size,
                )
        return output.images[0]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error("Out of memory during image generation. Your blog is preserved above.")
        else:
            st.error(f"Runtime error during image generation: {e}. Your blog is preserved above.")
        return None
    except Exception as e:
        st.error(f"Image generation failed: {e}. Your blog is preserved above.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def blog_to_txt(text: str) -> str:
    txt = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    txt = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", txt)
    return txt.strip()


def blog_to_docx(text: str, topic: str) -> bytes | None:
    try:
        from docx import Document
        from docx.shared import Pt
        doc = Document()
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                doc.add_heading(line[2:], level=1)
            elif line.startswith("## "):
                doc.add_heading(line[3:], level=2)
            elif line.startswith("### "):
                doc.add_heading(line[4:], level=3)
            else:
                para = doc.add_paragraph(line)
                para.style.font.size = Pt(11)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    except ImportError:
        return None
    except Exception:
        return None


def image_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ✍️ LlamaWrites")
    st.markdown("---")

    st.markdown("**Appearance**")
    theme_choice = st.radio(
        "Theme",
        options=["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    new_theme = "light" if "Light" in theme_choice else "dark"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")

    st.markdown("**Generation Mode**")
    mode_choice = st.radio(
        "Mode",
        options=["📝 Blog Only", "📝 + 🖼️ Blog + Image"],
        index=0 if st.session_state.gen_mode == "Blog Only" else 1,
        label_visibility="collapsed",
    )
    st.session_state.gen_mode = "Blog Only" if "Blog Only" in mode_choice else "Blog + Image"

    if st.session_state.gen_mode == "Blog Only":
        st.caption("Image model will NOT be loaded.")
    else:
        st.caption("Image model loads on demand after blog generation.")

    st.markdown("---")

    with st.expander("🤖 Model Info"):
        st.markdown(f"**Text:** `{TEXT_MODEL_ID}`")
        st.markdown(f"**Image:** `{IMAGE_MODEL_ID}`")
        st.markdown(f"**Device:** `{DEVICE.upper()}`")
        st.markdown("**Downloads (first run):**")
        st.markdown("- Text: ~2.6 GB")
        st.markdown("- Image: ~2 GB (Blog+Image only)")
        st.markdown("**CPU times:**")
        st.markdown("- Blog (300w): ~30-90 s")
        st.markdown("- Image (LCM 4 steps): ~2-5 min")

    st.markdown("---")

    with st.expander("💡 Tips"):
        st.markdown(
            "- **Blog Only** is significantly faster.\n"
            "- Image model is cached: loads once, reused every time.\n"
            "- Try **Listicle** or **How-To** for structured blogs.\n"
            "- Use **Regenerate Blog** to get a new variation."
        )

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="lw-header">
    <h1>✍️ LlamaWrites</h1>
    <p>AI-powered blog generation with matching imagery — runs fully locally, no API key required.</p>
</div>
""", unsafe_allow_html=True)

mode_emoji = "📝" if st.session_state.gen_mode == "Blog Only" else "📝+🖼️"
dev_icon = "🟢" if DEVICE == "cuda" else "🔵"
st.markdown(
    f'<span class="badge-device">{dev_icon} {DEVICE_LABEL}</span>'
    f'<span class="badge">{mode_emoji} {st.session_state.gen_mode}</span>',
    unsafe_allow_html=True,
)

if DEVICE == "cpu":
    st.markdown(
        '<div class="cpu-notice">⚠️ <strong>CPU mode.</strong> '
        'Blog: ~30-90 s. Image (LCM, 4 steps): ~2-5 min. '
        'Use <strong>Blog Only</strong> for fastest results.</div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📌 Blog Settings</div>', unsafe_allow_html=True)

col_topic, col_wc = st.columns([3, 1])
with col_topic:
    topic = st.text_input(
        "Blog Topic",
        placeholder="e.g. Artificial Intelligence, Climate Change, Hyderabad...",
        help="What should the blog be about?",
    )
with col_wc:
    word_count = st.number_input(
        "Word Count",
        min_value=100,
        max_value=1500,
        value=300,
        step=50,
        help="Target word count (100-1500). Actual output may vary by ~20%.",
    )

col_style, col_tone = st.columns(2)
with col_style:
    style_names = list(BLOG_STYLES.keys())
    selected_style = st.selectbox(
        "Blog Style",
        options=style_names,
        format_func=lambda s: f"{BLOG_STYLES[s]['icon']} {s}",
        help="Determines the structure and writing approach.",
    )
    st.caption(f"_{BLOG_STYLES[selected_style]['desc']}_")

with col_tone:
    selected_tone = st.selectbox(
        "Tone",
        options=list(TONES.keys()),
        help="Determines the voice and register of the writing.",
    )

st.markdown("")

btn_label = (
    "✨ Generate Blog"
    if st.session_state.gen_mode == "Blog Only"
    else "✨ Generate Blog + Image"
)
generate_btn = st.button(
    btn_label,
    type="primary",
    use_container_width=True,
    disabled=not topic.strip(),
)

if not topic.strip():
    st.caption("Enter a topic above to enable generation.")

# ──────────────────────────────────────────────────────────────────────────────
# Generation trigger
# ──────────────────────────────────────────────────────────────────────────────

if generate_btn and topic.strip():
    # Reset state
    st.session_state.blog_text       = None
    st.session_state.blog_image      = None
    st.session_state.blog_generated  = False
    st.session_state.image_generated = False
    st.session_state.image_error     = None
    st.session_state.last_topic      = topic.strip()
    st.session_state.last_style      = selected_style
    st.session_state.last_tone       = selected_tone
    st.session_state.last_wc         = int(word_count)

    # Load text model and generate blog
    generator = load_text_model()
    blog = generate_blog(
        st.session_state.last_topic,
        st.session_state.last_wc,
        st.session_state.last_style,
        st.session_state.last_tone,
        generator,
    )
    if blog:
        st.session_state.blog_text     = blog
        st.session_state.blog_generated = True

    # Generate image ONLY if Blog + Image mode selected
    if st.session_state.gen_mode == "Blog + Image" and st.session_state.blog_generated:
        st.info("Blog generated! Now loading image model and generating your image...")
        pipe, pipe_type = load_image_model()
        img = run_image_generation(
            st.session_state.last_topic,
            st.session_state.last_style,
            pipe,
            pipe_type or "",
        )
        if img:
            st.session_state.blog_image      = img
            st.session_state.image_generated = True
        else:
            st.session_state.image_error = "Image generation failed. See error above."

elif generate_btn and not topic.strip():
    st.warning("Please enter a blog topic before generating.")

# ──────────────────────────────────────────────────────────────────────────────
# Blog output
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.blog_generated and st.session_state.blog_text:
    blog_text   = st.session_state.blog_text
    actual_words = len(blog_text.split())
    t  = st.session_state.last_topic
    s  = st.session_state.last_style
    tn = st.session_state.last_tone
    wc = st.session_state.last_wc

    st.markdown('<div class="section-header">📄 Generated Blog</div>', unsafe_allow_html=True)

    st.markdown(
        f'<span class="badge">📌 {t}</span>'
        f'<span class="badge">{BLOG_STYLES[s]["icon"]} {s}</span>'
        f'<span class="badge">🎙️ {tn}</span>'
        f'<span class="badge">Requested: {wc} words</span>'
        f'<span class="badge">Actual: {actual_words} words</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.markdown('<div class="blog-content">', unsafe_allow_html=True)
    st.markdown(blog_text)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    col_rb, col_md, col_txt, col_docx = st.columns([2, 1, 1, 1])

    with col_rb:
        regen_blog_btn = st.button(
            "🔄 Regenerate Blog",
            help="Regenerate the blog with same settings. Does not regenerate the image.",
            use_container_width=True,
        )

    with col_md:
        st.download_button(
            label="⬇️ .md",
            data=blog_text,
            file_name=f"llamawrites_{t.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_txt:
        st.download_button(
            label="⬇️ .txt",
            data=blog_to_txt(blog_text),
            file_name=f"llamawrites_{t.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_docx:
        docx_bytes = blog_to_docx(blog_text, t)
        if docx_bytes:
            st.download_button(
                label="⬇️ .docx",
                data=docx_bytes,
                file_name=f"llamawrites_{t.replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        else:
            st.caption("DOCX: install python-docx")

    if regen_blog_btn:
        with st.spinner(f"Regenerating blog about '{t}'..."):
            generator = load_text_model()
            new_blog = generate_blog(t, wc, s, tn, generator)
            if new_blog:
                st.session_state.blog_text = new_blog
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Image output
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.blog_generated:
    if st.session_state.image_generated and st.session_state.blog_image is not None:
        img = st.session_state.blog_image
        t   = st.session_state.last_topic
        s   = st.session_state.last_style

        st.markdown('<div class="section-header">🖼️ Generated Image</div>', unsafe_allow_html=True)
        st.image(img, caption=f"AI-generated image for: {t}", use_container_width=True)

        col_ri, col_di, _ = st.columns([2, 1, 2])
        with col_ri:
            regen_img_btn = st.button(
                "🔄 Regenerate Image",
                help="Regenerate only the image. Blog is not changed.",
                use_container_width=True,
            )
        with col_di:
            st.download_button(
                label="⬇️ .png",
                data=image_to_bytes(img),
                file_name=f"llamawrites_{t.replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True,
            )

        if regen_img_btn:
            pipe, pipe_type = load_image_model()
            new_img = run_image_generation(t, s, pipe, pipe_type or "")
            if new_img:
                st.session_state.blog_image = new_img
                st.rerun()

    elif st.session_state.image_error:
        t = st.session_state.last_topic
        s = st.session_state.last_style
        st.warning(
            f"{st.session_state.image_error}  \n"
            "Your blog above is intact. You can retry image generation below."
        )
        if st.button("🔄 Retry Image Generation"):
            pipe, pipe_type = load_image_model()
            img = run_image_generation(t, s, pipe, pipe_type or "")
            if img:
                st.session_state.blog_image      = img
                st.session_state.image_generated = True
                st.session_state.image_error     = None
                st.rerun()

    elif st.session_state.gen_mode == "Blog + Image" and not st.session_state.image_generated:
        t = st.session_state.last_topic
        s = st.session_state.last_style
        st.markdown('<div class="section-header">🖼️ Image</div>', unsafe_allow_html=True)
        if st.button("🖼️ Generate Image Now"):
            pipe, pipe_type = load_image_model()
            img = run_image_generation(t, s, pipe, pipe_type or "")
            if img:
                st.session_state.blog_image      = img
                st.session_state.image_generated = True
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
fc = "#64748b" if st.session_state.theme == "dark" else "#94a3b8"
st.markdown(
    f"<div style='text-align:center;color:{fc};font-size:0.8rem;'>"
    "LlamaWrites v2.0 · Powered by Qwen2.5 & LCM Dreamshaper · 100% local · No API key required"
    "</div>",
    unsafe_allow_html=True,
)
