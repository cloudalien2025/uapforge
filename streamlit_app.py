# UAPForge v2.0 — 16:9 LANDSCAPE ONLY (1920x1080)
# ------------------------------------------------------------
# Real Photos (Google Places + Street View) + AI Render (OpenAI)
# Output: ALWAYS 1920x1080 (16:9) for CapCut landscape videos
#
# Requirements:
#   pip install streamlit requests pillow pandas openpyxl
#
# IMPORTANT (Google):
# Enable these Google APIs if using Real Photos mode:
#   • Places API (or Places API (New))
#   • Street View Static API

from __future__ import annotations

import base64
import io
import re
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image
import pandas as pd

# -----------------------------
# App constants
# -----------------------------
APP_NAME = "UAPForge v2.0 — 16:9 Landscape Image Generator"
OUTPUT_W, OUTPUT_H = 1920, 1080  # ✅ ONLY OUTPUT SIZE

# OpenAI base render sizes (we will crop/resize to 1920x1080)
# Use LANDSCAPE base by default for best results.
GPT_IMAGE_SIZES = ["1536x1024", "1024x1536", "1024x1024", "auto"]
DALLE3_SIZES = ["1792x1024", "1024x1792", "1024x1024"]  # if you ever choose dall-e-3

DEFAULT_WEBP_QUALITY = 82
MAX_KEYWORDS = 25

# Streamlit secrets convention:
# st.secrets["api_keys"]["OPENAI_API_KEY"]
# st.secrets["api_keys"]["GOOGLE_MAPS_API_KEY"]
SECRETS = st.secrets.get("api_keys", {})

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Candidate:
    title: str
    source: str
    preview_bytes: bytes
    license_note: str

# -----------------------------
# Helpers
# -----------------------------
def slugify(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "image"

def crop_resize_to_16x9(img: Image.Image, w: int = OUTPUT_W, h: int = OUTPUT_H) -> Image.Image:
    """Center-crop to 16:9 and resize to 1920x1080."""
    img = img.convert("RGB")
    target_ratio = w / h
    iw, ih = img.size

    if iw / ih > target_ratio:
        # too wide -> crop width
        new_w = int(ih * target_ratio)
        x0 = (iw - new_w) // 2
        box = (x0, 0, x0 + new_w, ih)
    else:
        # too tall -> crop height
        new_h = int(iw / target_ratio)
        y0 = (ih - new_h) // 2
        box = (0, y0, iw, y0 + new_h)

    return img.crop(box).resize((w, h), Image.LANCZOS)

def encode_webp_or_png(img: Image.Image, quality: int) -> Tuple[bytes, str, str]:
    """
    Tries WEBP first; falls back to PNG to prevent environment-specific crashes.
    Returns (bytes, ext, mime)
    """
    quality = int(max(40, min(100, quality)))
    buf = io.BytesIO()
    try:
        img.save(buf, format="WEBP", quality=quality)
        return buf.getvalue(), "webp", "image/webp"
    except Exception:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "png", "image/png"

def bytes_to_1920x1080(img_bytes: bytes, quality: int) -> Tuple[bytes, str, str]:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    final_img = crop_resize_to_16x9(img, OUTPUT_W, OUTPUT_H)
    return encode_webp_or_png(final_img, quality)

def safe_get_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {}

# -----------------------------
# Google (Real Photos)
# -----------------------------
GOOGLE_TEXTSEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_DETAILS = "https://maps.googleapis.com/maps/api/place/details/json"
GOOGLE_PHOTO = "https://maps.googleapis.com/maps/api/place/photo"
GOOGLE_STREET_META = "https://maps.googleapis.com/maps/api/streetview/metadata"
GOOGLE_STREET_IMG = "https://maps.googleapis.com/maps/api/streetview"

def google_textsearch_place(query: str, gmaps_key: str) -> Optional[dict]:
    r = requests.get(GOOGLE_TEXTSEARCH, params={"query": query, "key": gmaps_key}, timeout=30)
    if r.status_code != 200:
        return None
    data = safe_get_json(r)
    return (data.get("results") or [None])[0]

def google_place_details(place_id: str, gmaps_key: str) -> dict:
    r = requests.get(
        GOOGLE_DETAILS,
        params={"place_id": place_id, "fields": "name,geometry,photos", "key": gmaps_key},
        timeout=30,
    )
    if r.status_code != 200:
        return {}
    return (safe_get_json(r) or {}).get("result", {}) or {}

def google_photo_bytes(photo_ref: str, gmaps_key: str, max_w: int = 1600) -> Optional[bytes]:
    r = requests.get(
        GOOGLE_PHOTO,
        params={"photoreference": photo_ref, "maxwidth": max_w, "key": gmaps_key},
        timeout=30,
        allow_redirects=False,
    )
    # Google photo endpoint often responds with redirect to the actual image
    loc = r.headers.get("Location")
    if loc:
        img = requests.get(loc, timeout=30)
        if img.status_code == 200 and img.content:
            return img.content
    if r.status_code == 200 and r.content:
        return r.content
    return None

def streetview_bytes(lat: float, lng: float, gmaps_key: str, radius_m: int = 250,
                     size_w: int = 1280, size_h: int = 720) -> Optional[bytes]:
    meta = requests.get(
        GOOGLE_STREET_META,
        params={"location": f"{lat},{lng}", "radius": radius_m, "key": gmaps_key},
        timeout=20,
    )
    md = safe_get_json(meta)
    if md.get("status") != "OK":
        return None

    r = requests.get(
        GOOGLE_STREET_IMG,
        params={"location": f"{lat},{lng}", "radius": radius_m, "size": f"{size_w}x{size_h}", "key": gmaps_key},
        timeout=30,
    )
    if r.status_code == 200 and r.content:
        return r.content
    return None

def collect_real_photo_candidates(
    query: str,
    gmaps_key: str,
    use_places: bool,
    use_street: bool,
    sv_radius_m: int,
    max_places_photos: int,
) -> List[Candidate]:
    cands: List[Candidate] = []
    if not gmaps_key:
        return cands

    place = google_textsearch_place(query, gmaps_key)
    if not place or not place.get("place_id"):
        return cands

    details = google_place_details(place["place_id"], gmaps_key)
    title = details.get("name") or query
    loc = (details.get("geometry") or {}).get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")

    if use_places:
        photos = (details.get("photos") or [])[: max(1, int(max_places_photos))]
        for ph in photos:
            ref = ph.get("photo_reference")
            if not ref:
                continue
            raw = google_photo_bytes(ref, gmaps_key, max_w=1600)
            if raw:
                cands.append(Candidate(
                    title=f"Google Places Photo — {title}",
                    source="Google Maps contributor",
                    preview_bytes=raw,
                    license_note="License: Refer to Google Places Photo terms",
                ))

    if use_street and lat is not None and lng is not None:
        sv = streetview_bytes(lat, lng, gmaps_key, radius_m=int(sv_radius_m), size_w=1280, size_h=720)
        if sv:
            cands.append(Candidate(
                title=f"Google Street View — {title}",
                source="Google Street View",
                preview_bytes=sv,
                license_note="License: Refer to Google Street View terms",
            ))

    return cands

# -----------------------------
# OpenAI (AI Render)
# -----------------------------
OPENAI_BASE_URL = "https://api.openai.com/v1/images/generations"

def allowed_sizes_for_model(model: str) -> List[str]:
    if model.strip().lower() == "dall-e-3":
        return DALLE3_SIZES
    return GPT_IMAGE_SIZES

def pick_landscape_openai_size(model: str) -> str:
    sizes = allowed_sizes_for_model(model)
    # Prefer a landscape base
    for preferred in ["1536x1024", "1792x1024", "auto", "1024x1024"]:
        if preferred in sizes:
            return preferred
    return sizes[0]

def openai_generate_image_bytes(prompt: str, model: str, size: str, api_key: str, timeout_s: int = 120) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # request b64 when possible; fallback to url if needed
    payload = {"model": model, "prompt": prompt.strip(), "size": size, "n": 1, "response_format": "b64_json"}
    r = requests.post(OPENAI_BASE_URL, headers=headers, json=payload, timeout=timeout_s)

    # Some accounts/models may not accept response_format; retry without it
    if r.status_code == 400 and "response_format" in (r.text or ""):
        payload.pop("response_format", None)
        r = requests.post(OPENAI_BASE_URL, headers=headers, json=payload, timeout=timeout_s)

    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    jd = safe_get_json(r)
    data = (jd.get("data") or [])
    if not data:
        raise RuntimeError("OpenAI returned no image data.")

    item = data[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])

    if item.get("url"):
        img = requests.get(item["url"], timeout=60)
        if img.status_code == 200 and img.content:
            return img.content

    raise RuntimeError("OpenAI returned an empty image payload.")

def build_uap_prompt(keyword: str) -> str:
    """
    UAPForge style: cinematic, documentary still, no text/logos.
    You can harden this later with toggles (grain, timestamp overlay, etc.).
    """
    return (
        "Create a photorealistic, cinematic 16:9 landscape image (documentary still) based on: "
        f"'{keyword}'. "
        "Natural lighting, realistic textures, high detail, believable atmosphere. "
        "No text, no logos, no watermarks, no UI overlays. "
        "Compose with extra headroom and safe margins for captions."
    )

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Generates ONLY 1920×1080 landscape images for CapCut. Choose Real Photos, AI Render, or BOTH.")

mode = st.sidebar.radio("Mode", ["Both (Recommended)", "Real Photos Only", "AI Render Only", "Batch from Excel (Both)"], index=0)

st.sidebar.subheader("API Keys")
gmaps_key_input = st.sidebar.text_input("Google Maps/Places API key", type="password")
openai_key_input = st.sidebar.text_input("OpenAI API key", type="password")

gmaps_key = gmaps_key_input or SECRETS.get("GOOGLE_MAPS_API_KEY", "")
openai_key = openai_key_input or SECRETS.get("OPENAI_API_KEY", "")

st.sidebar.subheader("Output")
quality = st.sidebar.slider("Image quality (WEBP/PNG)", 40, 100, DEFAULT_WEBP_QUALITY)
st.sidebar.info(f"Output locked to: {OUTPUT_W}x{OUTPUT_H} (16:9)")

# Real photo controls
st.sidebar.subheader("Real Photos Sources")
use_places = st.sidebar.checkbox("Google Places Photos", value=True)
use_street = st.sidebar.checkbox("Google Street View", value=True)
max_places_photos = st.sidebar.number_input("Max Places photos per query", 1, 12, 6)
sv_radius_m = st.sidebar.slider("Street View radius (meters)", 25, 500, 250)

# AI controls
st.sidebar.subheader("AI Render Settings")
model = st.sidebar.selectbox("Model", ["gpt-image-1", "dall-e-3"], index=0)
ai_images_per_keyword = st.sidebar.number_input("AI images per keyword", 1, 6, 1)
real_images_per_keyword = st.sidebar.number_input("Real images to include per keyword", 1, 6, 2)

with st.sidebar.expander("Debug", expanded=False):
    show_debug = st.toggle("Show errors/details", value=True)

# -----------------------------
# Keyword input (modes except Excel)
# -----------------------------
if mode != "Batch from Excel (Both)":
    st.subheader("Input")
    keywords_text = st.text_area(
        "Paste keywords (one per line). For real photos, include a place name + city/state for best results.",
        height=140,
        placeholder="Trinity Site, New Mexico\nPhoenix Lights sighting location, Phoenix Arizona\nVandenberg Space Force Base coastline, California\n"
    )
    keywords = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]

    colA, colB = st.columns([1, 1])
    run = colA.button("Generate", type="primary")
    clear = colB.button("Clear")

    if clear:
        st.rerun()

    if run:
        if not keywords:
            st.warning("Paste at least one keyword.")
            st.stop()

        if len(keywords) > MAX_KEYWORDS:
            st.warning(f"Using first {MAX_KEYWORDS} keywords to avoid rate limits.")
            keywords = keywords[:MAX_KEYWORDS]

        need_real = mode in ("Both (Recommended)", "Real Photos Only")
        need_ai = mode in ("Both (Recommended)", "AI Render Only")

        if need_real and not gmaps_key:
            st.error("Google Maps/Places API key is required for Real Photos.")
            st.stop()

        if need_ai and not openai_key:
            st.error("OpenAI API key is required for AI Render.")
            st.stop()

        openai_size = pick_landscape_openai_size(model)

        zip_buf = io.BytesIO()
        zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

        previews: List[Tuple[str, bytes, str, str]] = []  # (caption, bytes, filename, mime)
        errors: List[Tuple[str, str]] = []

        total_steps = len(keywords)
        prog = st.progress(0.0)

        for i, kw in enumerate(keywords, start=1):
            st.markdown(f"### {kw}")

            # ----- REAL PHOTOS -----
            if need_real:
                try:
                    cands = collect_real_photo_candidates(
                        query=kw,
                        gmaps_key=gmaps_key,
                        use_places=use_places,
                        use_street=use_street,
                        sv_radius_m=int(sv_radius_m),
                        max_places_photos=int(max_places_photos),
                    )
                    if not cands:
                        st.info("Real Photos: no candidates found.")
                    else:
                        chosen = cands[: int(real_images_per_keyword)]
                        for j, c in enumerate(chosen, start=1):
                            out_bytes, ext, mime = bytes_to_1920x1080(c.preview_bytes, quality)
                            fn = f"{slugify(kw)}__real_{j}.{ext}"
                            zf.writestr(fn, out_bytes)

                            st.image(out_bytes, caption=f"REAL: {c.title}", use_container_width=True)
                            st.caption(f"Credit: {c.source}")
                            st.caption(c.license_note)

                            st.download_button(
                                "Download REAL",
                                data=out_bytes,
                                file_name=fn,
                                mime=mime,
                                key=f"dl_real_{i}_{j}_{fn}",
                            )

                            previews.append((f"REAL — {kw}", out_bytes, fn, mime))
                except Exception as e:
                    msg = f"Real Photos error: {e}"
                    errors.append((kw, msg))
                    st.error(msg)
                    if show_debug:
                        st.code(repr(e))

            # ----- AI RENDER -----
            if need_ai:
                try:
                    for j in range(1, int(ai_images_per_keyword) + 1):
                        # light variation if generating multiple
                        if j == 1:
                            ai_kw = kw
                        else:
                            ai_kw = f"{kw} — alternate cinematic angle {j}"

                        prompt = build_uap_prompt(ai_kw)
                        raw = openai_generate_image_bytes(prompt, model=model, size=openai_size, api_key=openai_key)
                        out_bytes, ext, mime = bytes_to_1920x1080(raw, quality)
                        fn = f"{slugify(kw)}__ai_{j}.{ext}"
                        zf.writestr(fn, out_bytes)

                        st.image(out_bytes, caption=f"AI: {ai_kw}", use_container_width=True)
                        st.download_button(
                            "Download AI",
                            data=out_bytes,
                            file_name=fn,
                            mime=mime,
                            key=f"dl_ai_{i}_{j}_{fn}",
                        )

                        previews.append((f"AI — {kw}", out_bytes, fn, mime))
                except Exception as e:
                    msg = f"AI Render error: {e}"
                    errors.append((kw, msg))
                    st.error(msg)
                    if show_debug:
                        st.code(repr(e))

            prog.progress(i / total_steps)

        zf.close()
        zip_buf.seek(0)

        st.markdown("---")
        st.subheader("Download everything")
        st.download_button(
            "Download ZIP (All 1920x1080)",
            data=zip_buf.getvalue(),
            file_name="uapforge_1920x1080.zip",
            mime="application/zip",
            key="zip_all",
        )

        if errors:
            st.subheader("Errors")
            for kw, msg in errors:
                st.error(f"{kw}: {msg}")

# -----------------------------
# Batch from Excel (Both)
# -----------------------------
else:
    st.subheader("Batch from Excel (Both)")
    st.write("Upload an `.xlsx` file with a column of keywords/locations. We'll generate REAL + AI 1920x1080 for each row.")

    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if file:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Unable to read Excel: {e}")
            st.stop()

        if df.empty:
            st.warning("Spreadsheet is empty.")
            st.stop()

        default_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("keyword", "query", "place", "name", "location"):
                default_col = c
                break

        colname = st.selectbox(
            "Select the column containing the keyword/location queries:",
            list(df.columns),
            index=(list(df.columns).index(default_col) if default_col in df.columns else 0),
        )

        context_hint = st.text_input("Optional context appended to each query (e.g., 'USA' or 'Arizona')", "")

        start = st.button("Run Batch", type="primary")

        if start:
            if not gmaps_key:
                st.error("Google Maps/Places API key is required for Real Photos.")
                st.stop()
            if not openai_key:
                st.error("OpenAI API key is required for AI Render.")
                st.stop()

            rows = df[colname].astype(str).tolist()
            rows = [r.strip() for r in rows if r and str(r).strip()]
            if not rows:
                st.warning("No usable rows in that column.")
                st.stop()

            openai_size = pick_landscape_openai_size(model)

            zip_buf = io.BytesIO()
            zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

            prog = st.progress(0.0)
            created = 0

            for i, raw in enumerate(rows, start=1):
                q = raw
                if context_hint:
                    q = f"{q}, {context_hint}"

                # Real candidates
                cands = collect_real_photo_candidates(
                    query=q,
                    gmaps_key=gmaps_key,
                    use_places=use_places,
                    use_street=use_street,
                    sv_radius_m=int(sv_radius_m),
                    max_places_photos=int(max_places_photos),
                )
                chosen = cands[: int(real_images_per_keyword)]

                for j, c in enumerate(chosen, start=1):
                    try:
                        out_bytes, ext, _mime = bytes_to_1920x1080(c.preview_bytes, quality)
                        fn = f"{slugify(raw)}__real_{i}_{j}.{ext}"
                        zf.writestr(fn, out_bytes)
                        created += 1
                    except Exception:
                        pass

                # AI images
                for j in range(1, int(ai_images_per_keyword) + 1):
                    try:
                        ai_kw = raw if j == 1 else f"{raw} — alternate cinematic angle {j}"
                        prompt = build_uap_prompt(ai_kw)
                        raw_img = openai_generate_image_bytes(prompt, model=model, size=openai_size, api_key=openai_key)
                        out_bytes, ext, _mime = bytes_to_1920x1080(raw_img, quality)
                        fn = f"{slugify(raw)}__ai_{i}_{j}.{ext}"
                        zf.writestr(fn, out_bytes)
                        created += 1
                    except Exception:
                        pass

                prog.progress(i / len(rows))

            zf.close()
            zip_buf.seek(0)

            st.success(f"Done. Created {created} files (all 1920x1080).")
            st.download_button(
                "Download ZIP",
                data=zip_buf.getvalue(),
                file_name="uapforge_batch_1920x1080.zip",
                mime="application/zip",
                key="zip_batch",
            )
