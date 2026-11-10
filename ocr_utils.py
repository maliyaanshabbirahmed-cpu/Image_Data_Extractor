"""Image preprocessing, OCR, conservative structuring and export helpers."""
import io
import re
import time

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

LANGUAGES = {
    "English": "eng",
    "Urdu": "urd",
    "Arabic": "ara",
    "English + Urdu": "eng+urd",
    "English + Arabic": "eng+ara",
    "English + Urdu + Arabic": "eng+urd+ara",
}


def extract_pdf_page(data: bytes) -> Image.Image:
    """Render the first PDF page to an RGB PIL image."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF support is unavailable. Install PyMuPDF.") from exc
    try:
        document = fitz.open(stream=data, filetype="pdf")
        if document.page_count < 1:
            raise ValueError("The PDF contains no pages.")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc


def enhance_image(image: Image.Image) -> Image.Image:
    """Denoise, normalize contrast, deskew slightly and binarize."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    array = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    height, width = array.shape
    longest = max(height, width)
    if longest < 1800:
        scale = min(2.0, 1800.0 / longest)
        array = cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    array = cv2.fastNlMeansDenoising(array, None, 10, 7, 21)
    array = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(array)

    try:
        coordinates = np.column_stack(np.where(array < 220))
        if len(coordinates) > 100:
            angle = cv2.minAreaRect(coordinates)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if 0.4 < abs(angle) < 12:
                h, w = array.shape
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                array = cv2.warpAffine(
                    array, matrix, (w, h), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
    except Exception:
        # Deskew is best-effort; OCR can continue with the enhanced image.
        pass

    _, binary = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def extract_with_tesseract(image: Image.Image, language: str):
    """Return recognized text, token dataframe and mean OCR confidence."""
    try:
        available = set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise FileNotFoundError(
            "Tesseract OCR was not found. On Streamlit Cloud, confirm packages.txt is present; "
            "locally, install Tesseract and add it to PATH."
        ) from exc

    missing = [code for code in language.split("+") if code not in available]
    if missing:
        raise RuntimeError(
            "Required Tesseract language pack(s) are missing: " + ", ".join(missing) +
            ". Install the matching tesseract language packages or choose another language."
        )

    config = "--oem 3 --psm 3"
    text = pytesseract.image_to_string(image, lang=language, config=config).strip()
    data = pytesseract.image_to_data(
        image, lang=language, config=config, output_type=Output.DATAFRAME
    )
    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data["conf"] = pd.to_numeric(data["conf"], errors="coerce")
    valid = data[data["conf"] >= 0]
    confidence = float(valid["conf"].mean()) if not valid.empty else 0.0
    return text, data, confidence


def text_to_dataframe(text: str) -> pd.DataFrame:
    """Structure OCR output conservatively; do not infer missing cell values."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return pd.DataFrame({"Extracted text": []})

    rows = []
    for line in lines:
        if "|" in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
        elif "\t" in line:
            parts = [part.strip() for part in line.split("\t")]
        else:
            parts = [line]
        rows.append(parts)

    width = max(len(row) for row in rows)
    if width > 1:
        rows = [row + [""] * (width - len(row)) for row in rows]
        return pd.DataFrame(rows, columns=[f"Column {i + 1}" for i in range(width)])

    pairs = []
    for line in lines:
        match = re.match(r"^\s*([^:：]{1,80})\s*[:：]\s*(.*?)\s*$", line)
        if match:
            pairs.append((match.group(1).strip(), match.group(2).strip()))
    if len(pairs) >= 2:
        return pd.DataFrame(pairs, columns=["Field", "Value"])

    return pd.DataFrame({"Extracted text": lines})


def build_quality_report(filename, language, text, ocr_rows, dataframe, confidence, elapsed):
    words = len(ocr_rows)
    number_tokens = 0
    if words:
        number_tokens = int(ocr_rows["text"].astype(str).str.contains(r"\d", regex=True).sum())
    return {
        "File": filename,
        "OCR language": language,
        "Words detected": words,
        "Number-containing tokens": number_tokens,
        "Rows extracted": len(dataframe),
        "Columns extracted": len(dataframe.columns),
        "Average OCR confidence (%)": round(float(confidence), 1),
        "Processing time (seconds)": round(float(elapsed), 2),
        "Warnings": "Handwriting, complex tables and mathematical notation may be misrecognized.",
    }


def build_excel(dataframe, recognized_text, report) -> bytes:
    """Create a formatted workbook with data, raw OCR and quality report sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Extracted Data")
        pd.DataFrame({"Recognized text": recognized_text.splitlines()}).to_excel(
            writer, index=False, sheet_name="OCR Text"
        )
        pd.DataFrame([report]).to_excel(writer, index=False, sheet_name="Quality Report")

        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

        worksheet = writer.sheets["Extracted Data"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        header_fill = PatternFill("solid", fgColor="D6B354")
        header_font = Font(bold=True, color="17130A")
        bottom = Side(style="thin", color="B58A2A")
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = Border(bottom=bottom)
        for column in worksheet.columns:
            letter = column[0].column_letter
            max_length = max((len(str(cell.value or "")) for cell in column), default=10)
            worksheet.column_dimensions[letter].width = min(max(max_length + 3, 14), 48)
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        for sheet in (writer.sheets["OCR Text"], writer.sheets["Quality Report"]):
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
            for column in sheet.columns:
                letter = column[0].column_letter
                max_length = max((len(str(cell.value or "")) for cell in column), default=10)
                sheet.column_dimensions[letter].width = min(max(max_length + 3, 14), 48)

    output.seek(0)
    return output.getvalue()
