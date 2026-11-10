import io
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from ocr_utils import (
    LANGUAGES, enhance_image, extract_pdf_page, extract_with_tesseract,
    text_to_dataframe, build_excel, build_quality_report,
)

APP_TITLE = "Image Data Extractor"
MAX_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
.stApp {background: radial-gradient(circle at 12% 0%, #28200e 0, #111 38%, #090909 100%); color:#f4f0e8;}
section[data-testid="stSidebar"] {background:#11100d;border-right:1px solid #66501e;}
h1,h2,h3 {color:#e7c46a !important;}
div[data-testid="stMetric"] {background:#171510;border:1px solid #59451b;padding:14px;border-radius:12px;}
.stButton>button,.stDownloadButton>button {background:linear-gradient(135deg,#d6b354,#9c7626);color:#100e08;border:0;border-radius:9px;font-weight:700;}
.stButton>button:hover,.stDownloadButton>button:hover {background:#f0d27e;color:#100e08;}
[data-testid="stFileUploader"] {background:#15130f;border:1px dashed #b18b37;border-radius:12px;padding:12px;}
.gold-card {background:#171510;border:1px solid #59451b;border-radius:14px;padding:18px;margin:8px 0;}
.muted {color:#b9b2a2;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

def get_source_image(uploaded_file):
    """Load an uploaded image or the first page of an uploaded PDF."""
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError("The file exceeds the 10 MB upload limit.")
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_page(raw), "PDF (first page only)"
    try:
        image = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(image).convert("RGB"), "Image"
    except Exception as exc:
        raise ValueError(f"Could not open the image: {exc}") from exc

st.title("Image Data Extractor")
st.markdown(
    '<div class="muted">Convert image and document text into structured CSV and Excel files using local OCR.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Processing settings")
    language_label = st.selectbox("OCR language", list(LANGUAGES.keys()))
    use_enhancement = st.checkbox("Enhance image before OCR", value=True)
    st.caption("Runs locally with Tesseract. No API key or paid service is required.")
    st.divider()
    st.markdown("**Supported files**")
    st.write("JPG · JPEG · PNG · PDF")
    st.caption("10 MB maximum. One file per extraction. PDFs: first page only.")

st.markdown(
    '<div class="gold-card"><b>Upload a document</b><br>'
    '<span class="muted">Upload one image/PDF or capture a photo using your device camera.</span></div>',
    unsafe_allow_html=True,
)
col_upload, col_camera = st.columns(2)
with col_upload:
    upload = st.file_uploader("Choose one image or PDF", type=["jpg", "jpeg", "png", "pdf"])
with col_camera:
    camera = st.camera_input("Take a photograph")

source = camera if camera is not None else upload
if source is not None:
    try:
        original, source_type = get_source_image(source)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.markdown("### Image preview and optional crop")
    left, right = st.columns(2)
    with left:
        st.image(original, caption="Original image", use_container_width=True)
    with right:
        st.caption("Optional: select a region to extract. Leave the full image selected if you want everything.")
        try:
            from streamlit_cropper import st_cropper
            cropped = st_cropper(original, realtime_update=True, box_color="#d6b354", aspect_ratio=None, return_type="image")
            selected_image = cropped.convert("RGB") if cropped is not None else original
        except Exception:
            selected_image = original
            st.info("The crop tool is unavailable in this environment. The full image will be processed.")

    if st.button("Extract data", type="primary", use_container_width=True):
        started = time.perf_counter()
        try:
            with st.spinner("Preparing image and recognizing text..."):
                processed = enhance_image(selected_image) if use_enhancement else selected_image
                text, ocr_rows, confidence = extract_with_tesseract(processed, LANGUAGES[language_label])
                dataframe = text_to_dataframe(text)
                elapsed = round(time.perf_counter() - started, 2)
                report = build_quality_report(
                    filename=getattr(source, "name", "camera_capture"),
                    language=language_label,
                    text=text,
                    ocr_rows=ocr_rows,
                    dataframe=dataframe,
                    confidence=confidence,
                    elapsed=elapsed,
                )
                st.session_state["extraction_result"] = {
                    "text": text,
                    "dataframe": dataframe,
                    "report": report,
                    "csv": dataframe.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    "excel": build_excel(dataframe, text, report),
                    "original": original,
                    "processed": processed,
                }
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")

if "extraction_result" in st.session_state:
    result = st.session_state["extraction_result"]
    report = result["report"]
    st.divider()
    st.markdown("## Extraction results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words detected", report["Words detected"])
    m2.metric("Rows extracted", report["Rows extracted"])
    m3.metric("Columns", report["Columns extracted"])
    m4.metric("Average confidence", f'{report["Average OCR confidence (%)"]}%')

    image_col, result_col = st.columns([1, 1.25])
    with image_col:
        st.image(result["original"], caption="Original", use_container_width=True)
        st.image(result["processed"], caption="OCR-ready image", use_container_width=True)
    with result_col:
        st.markdown("### Extracted data preview")
        st.dataframe(result["dataframe"], use_container_width=True, hide_index=True)
        st.markdown("### Recognized text")
        st.text_area("OCR text", result["text"], height=180, disabled=True)

    st.markdown("### Quality report")
    st.dataframe(pd.DataFrame([report]), use_container_width=True, hide_index=True)
    if report["Average OCR confidence (%)"] < 60:
        st.warning("Low OCR confidence. Results may be incomplete or inaccurate; compare them with the source.")
    if not result["text"].strip():
        st.warning("No text was recognized. Try a clearer, better-lit image or another language setting.")

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(report["File"]).stem) or "extraction"
    st.markdown("### Download")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download CSV", result["csv"], f"{safe_name}.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button(
            "Download Excel", result["excel"], f"{safe_name}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    retention = st.radio(
        "Keep this result in the current session?",
        ["Keep until session ends", "Clear result now"],
        horizontal=True,
    )
    if retention == "Clear result now":
        del st.session_state["extraction_result"]
        st.rerun()

st.divider()
st.caption(
    "Privacy: this app does not save extraction history or intentionally store uploaded files permanently. "
    "Hosting providers may retain operational logs or temporary data. Verify important values against the original."
)
