# app.py — UAPForge (AI Render only) v1.2 (Streamlit Cloud hardened)
import base64
import io
import time
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
import streamlit as st
from PIL import Image

APP_TITLE = "UAPForge — AI Render (CapCut Ready)"

GPT_IMAGE_SIZES = ["1024x1536", "1536x1024", "1024x1024", "auto"]
DALLE3_SIZES = ["1024x1024", "1792x1024", "1024x1792"]

DEFAULT_WEBP_QUALITY = 82
MAX_KEYWORDS = 25

CAPCUT_PRESETS = {
    "Vertical 9:16 (1080x1920) - Shorts/Reels": (1080, 1920),
    "Landscape 16:9 (1920x1080) - YouTube/Doc": (1920, 1080),
    "4K Landscape 16:9 (3840x2160) - Ultra": (3840, 2160),
}

# ---------- helpers ----------
def slugify(s: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in (s or "").strip().lower()).strip("-")
    return out or "image"

def center_crop_and_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_size = (int(src_w * scale), int(src_h * scale))
    img = img.resize(new_size, Image.LANCZOS)

    left = (img.width - target_w) // 2
    top = (img.height - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))

def encode_image_bytes(img: Image.Image, quality: int) -> Tuple[bytes, str, str]:
    """
    Returns: (bytes, file_ext, mime)
    Tries WEBP first; falls back to PNG if WEBP isn't supported in the runtime.
    """
    buf = io.BytesIO()
    try:
        # Keep this minimal to avoid codec/option issues on Streamlit Cloud
        img.save(buf, format="WEBP", quality=int(quality))
        return buf.getvalue(), "webp", "image/webp"
    except Exception:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "png", "image/png"

def allowed_sizes_for_model(model: str):
    return DALLE3_SIZES if model.strip().lower() == "dall-e-3" else GPT_IMAGE_SIZES

def pick_base_size(model: str, target_w: int, target_h: int) -> str:
    sizes = allowed_sizes_for_model(model)

    if target_h > target_w:
        return "1024x1536" if "1024x1536" in sizes else ("1024x1792" if "1024x1792" in sizes else sizes[0])
    if target_w > target_h:
        return "1536x1024" if "1536x1024" in sizes else ("1792x1024" if "1792x1024" in sizes else sizes[0])
    return "1024x1024" if "1024x1024" in sizes else sizes[0]

# ---------- OpenAI Client ----------
@dataclass
class OpenAIClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"

    def generate_image(self, prompt: str, model: str, size: str, timeout: int = 120) -> requests.Response:
        url = f"{self.base_url}/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "prompt": prompt.strip(), "size": size, "n": 1}
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

def _bytes_from_openai_item(item: dict, timeout: int = 120) -> Optional[bytes]:
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        r = requests.get(item["url"], timeout=timeout)
        r.raise_for_status()
        return r.content
    return None

def generate_capcut_image_bytes(
    client: OpenAIClient,
    prompt: str,
    model: str,
    openai_size: str,
    target_w: int,
    target_h: int,
    quality: int,
    max_retries: int = 4,
    timeout_s: int = 120,
) -> Tuple[Optional[bytes], Optional[str], Optional[str], Optional[str]]:
    """
    Returns (img_bytes, err, debug, ext)
    """
    delay = 1.2
    last_debug = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.generate_image(prompt=prompt, model=model, size=openai_size, timeout=timeout_s)

            if resp.status_code == 200:
                data = resp.json()
                if not data.get("data"):
                    return None, "OpenAI returned no image data.", None, None
                raw = _bytes_from_openai_item(data["data"][0], timeout=timeout_s)
                if not raw:
                    return None, "OpenAI returned an empty image payload.", None, None

                img = Image.open(io.BytesIO(raw)).convert("RGB")
                final_img = center_crop_and_resize(img, target_w, target_h)
                out_bytes, ext, _mime = encode_image_bytes(final_img, quality)
                return out_bytes, None, None, ext

            try:
                j = resp.json()
                last_debug = j.get("error", {}).get("message", resp.text)
            except Exception:
                last_debug = resp.text

            transient = any(x in (last_debug or "").lower() for x in [
                "rate limit", "429", "timeout", "timed out", "gateway", "overloaded", "502", "503", "504"
            ])
            if transient:
                time.sleep(delay)
                delay *= 1.8
                continue

            return None, f"OpenAI error HTTP {resp.status_code}", last_debug, None

        except Exception as e:
            last_debug = str(e)
            transient = any(x in last_debug.lower() for x in [
                "rate limit", "429", "timeout", "timed out", "gateway", "overloaded", "502", "503", "504"
            ])
            if transient:
                time.sleep(delay)
                delay *= 1.8
                continue
            return None, "Request failed (exception).", last_debug, None

    return None, f"Failed after {max_retries} retries.", last_debug, None

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("AI-only image generation. Outputs are cropped/resized for CapCut.")

# show versions to help diagnose environment
with st.expander("Environment (for debugging)", expanded=False):
    st.write("Streamlit:", st.__version__)
    st.write("Requests:", requests.__version__)
    st.write("Pillow:", Image.__module__)

# Secrets first
try:
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    secret_key = ""

st.sidebar.header("API Key")
openai_key = st.sidebar.text_input("OpenAI API key", type="password", value=secret_key)

st.sidebar.header("Output")
preset_label = st.sidebar.selectbox("CapCut preset", list(CAPCUT_PRESETS.keys()), index=0)
target_w, target_h = CAPCUT_PRESETS[preset_label]
quality = st.sidebar.slider("Quality (WEBP/PNG)", 40, 100, DEFAULT_WEBP_QUALITY)

with st.sidebar.expander("Advanced", expanded=False):
    model = st.selectbox("Model", ["gpt-image-1", "dall-e-3"], index=0)
    sizes_for_model = allowed_sizes_for_model(model)
    size_mode = st.radio("Base size mode", ["Auto", "Force"], index=0)
    forced_size = st.selectbox("Forced base size", sizes_for_model, index=0)
    show_debug = st.toggle("Show debug output", value=True)

st.subheader("Input")
keywords_text = st.text_area(
    "One prompt per line",
    height=160,
    placeholder="UAP hovering over rural highway at night, cinematic, realistic\n1950s radar room, tense mood, grainy documentary still\n..."
)
keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]

go = st.button("Generate images", type="primary")
st.info(f"Preset output: {target_w}x{target_h}")

if go:
    if not openai_key:
        st.error("Add your OpenAI API key in the sidebar (or set Streamlit secret OPENAI_API_KEY).")
        st.stop()
    if not keywords:
        st.warning("Add at least one prompt.")
        st.stop()

    if len(keywords) > MAX_KEYWORDS:
        st.warning(f"Generating first {MAX_KEYWORDS} prompts to avoid rate limits.")
        keywords = keywords[:MAX_KEYWORDS]

    base_size = pick_base_size(model, target_w, target_h) if size_mode == "Auto" else forced_size
    client = OpenAIClient(api_key=openai_key)

    zip_buffer = io.BytesIO()
    zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

    previews = []
    errors = []

    for i, kw in enumerate(keywords, start=1):
        spinner_label = f"Generating {i}/{len(keywords)}: {kw}"
        with st.spinner(spinner_label):
            img_bytes, err, debug, ext = generate_capcut_image_bytes(
                client=client,
                prompt=kw,
                model=model,
                openai_size=base_size,
                target_w=target_w,
                target_h=target_h,
                quality=quality,
            )

        if err:
            errors.append((kw, err, debug))
            st.error(f"{kw}: {err}")
            if show_debug and debug:
                st.code(debug)
            continue

        fname = f"{slugify(kw)}.{ext}"
        zf.writestr(fname, img_bytes)
        previews.append((kw, img_bytes, fname, ext))
        st.success(f"Done: {fname}")

    zf.close()
    zip_buffer.seek(0)

    if previews:
        st.subheader("Previews")
        cols = st.columns(3)
        for idx, (cap, bts, fname, ext) in enumerate(previews):
            with cols[idx % 3]:
                st.image(bts, caption=cap, use_container_width=True)
                mime = "image/webp" if ext == "webp" else "image/png"
                st.download_button("Download", data=bts, file_name=fname, mime=mime, key=f"dl-{idx}")

    if errors:
        st.subheader("Errors")
        for kw, err, debug in errors:
            st.error(f"{kw}: {err}")
            if show_debug and debug:
                st.code(debug)

    st.subheader("Download all")
    st.download_button(
        "Download ZIP",
        data=zip_buffer.getvalue(),
        file_name="uapforge_images.zip",
        mime="application/zip",
        key="zip-all",
    )
