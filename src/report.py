"""Vector PDF pre-analysis report generator for MamNexa AI (Phase III).

Renders the guardrail-checked explanation bundle (see ``explain.explain_case``)
into a single ``fpdf2`` vector PDF: the Model Malignancy Suspicion Index and its
cautious interpretation, the Grad-CAM attention overlay, the numbered
AI-Identified Suspicious Areas, a per-region table, and a prominent
research-only disclaimer. The document is a *pre-analysis research aid*, not a
diagnosis.

Design notes
------------
* **Guardrails first.** Every human-facing string that will land in the PDF is
  swept against :data:`config.FORBIDDEN_PHRASES` *before* any drawing happens,
  so a wording regression raises here instead of shipping a bad report.
* **No TensorFlow.** This module only needs the vocabulary list and the bundle
  dict, so it stays lightweight and fast to test.
* **Vector + latin-1.** Core fonts (Helvetica) keep the PDF vector and small.
  Text is coerced to latin-1-safe ASCII because core fonts cannot encode
  arbitrary Unicode; all mandated copy is ASCII already, so this only guards
  against stray characters in technical metadata.
* **Cursor discipline.** ``multi_cell(w=0, align="C")`` in fpdf2 leaves the
  cursor at the right margin; every text block therefore passes
  ``new_x="LMARGIN"`` so the following block starts with full width.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fpdf import FPDF
from PIL import Image

from .config import (
    DECISION_DISCLAIMER,
    FORBIDDEN_PHRASES,
    LICENSE_NOTICE,
    REPORT_TITLE,
    SUSPICION_INDEX_NAME,
)
from .model_status import ModelStatus, overall_mode

# A4 portrait geometry (mm).
_MARGIN = 15
_PAGE_WIDTH = 210
_USABLE_WIDTH = _PAGE_WIDTH - 2 * _MARGIN  # 180


# ---------------------------------------------------------------------------
# Image helpers (also reused by the Streamlit app for object storage)
# ---------------------------------------------------------------------------
def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a float [0,1] or integer image to uint8 [0,255]; squeeze (H,W,1)."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)  # assume [0, 1]
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def _arr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a grayscale/RGB(A) numpy image to a PIL image for embedding."""
    a = _normalize_to_uint8(arr)
    if a.ndim == 2:
        return Image.fromarray(a, mode="L")
    if a.ndim == 3 and a.shape[-1] == 3:
        return Image.fromarray(a, mode="RGB")
    if a.ndim == 3 and a.shape[-1] == 4:
        return Image.fromarray(a, mode="RGBA").convert("RGB")
    raise ValueError(f"Unsupported image shape for PDF embedding: {a.shape}")


def np_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a numpy image as PNG bytes (used for the report and S3 upload)."""
    buf = io.BytesIO()
    _arr_to_pil(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Text hygiene + guardrail sweep
# ---------------------------------------------------------------------------
_UNICODE_FALLBACKS = {
    "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "µ": "u",
    "×": "x", "−": "-",
}


def _ascii(text: Any) -> str:
    """Make text safe for a latin-1 core font (replace/drop non-encodable chars)."""
    s = str(text)
    for bad, good in _UNICODE_FALLBACKS.items():
        s = s.replace(bad, good)
    return s.encode("latin-1", "ignore").decode("latin-1")


def _lesion_rows(lesions: list[dict[str, Any]]) -> list[list[str]]:
    """Flatten the lesion table into ASCII cell strings for rendering."""
    rows: list[list[str]] = []
    for les in lesions:
        cx, cy = les.get("centroid", (0, 0))
        bx, by, bw, bh = les.get("bbox", (0, 0, 0, 0))
        rows.append(
            [
                f"#{les.get('region_id', '?')}",
                _ascii(les.get("label", "")),
                f"{float(les.get('area_px', 0)):.1f}",
                f"({cx:.0f}, {cy:.0f})",
                f"{bx}, {by}, {bw}, {bh}",
            ]
        )
    return rows


def _collect_report_text(
    bundle: dict[str, Any],
    safe_meta: dict[str, Any] | None,
    case_id: str | None,
    created_at: str,
    model_statuses: list[ModelStatus] | None = None,
    data_provenance: str | None = None,
) -> list[str]:
    """Gather every string that will be printed, for the pre-render guardrail sweep."""
    interp = bundle.get("interpretation", {}) or {}
    texts: list[str] = [
        REPORT_TITLE,
        LICENSE_NOTICE,
        DECISION_DISCLAIMER,
        str(bundle.get("disclaimer", "")),
        str(case_id or ""),
        created_at,
        str(data_provenance or ""),
    ]
    texts.extend(str(v) for v in interp.values())
    texts.extend(str(les.get("label", "")) for les in bundle.get("lesions", []))
    for status in model_statuses or []:
        texts.extend([status.short(), status.banner])
    if model_statuses:
        texts.append(overall_mode(model_statuses))
    if safe_meta:
        texts.extend(f"{k}: {v}" for k, v in safe_meta.items())
    return texts


def _assert_report_text_safe(texts: list[str]) -> None:
    """Fail loud if any report text contains forbidden diagnostic phrasing."""
    blob = " ".join(texts).lower()
    hits = [p for p in FORBIDDEN_PHRASES if p in blob]
    if hits:
        raise ValueError(f"Guardrail violation - forbidden phrasing in report: {hits}")


# ---------------------------------------------------------------------------
# PDF document
# ---------------------------------------------------------------------------
class _ReportPDF(FPDF):
    """FPDF with a fixed research/licensing footer on every page."""

    def footer(self) -> None:  # noqa: D401 - fpdf2 hook
        self.set_y(-14)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.multi_cell(
            0, 4, _ascii(f"{LICENSE_NOTICE}  |  Page {self.page_no()}"),
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.set_text_color(0, 0, 0)


def _para(
    pdf: _ReportPDF,
    text: str,
    *,
    h: float = 6.0,
    size: int = 11,
    style: str = "",
    align: str = "L",
    fill: bool = False,
    border: int = 0,
) -> None:
    """Full-width text block that always leaves the cursor at the left margin."""
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(
        0, h, _ascii(text), align=align, fill=fill, border=border,
        new_x="LMARGIN", new_y="NEXT",
    )


def _heading(pdf: _ReportPDF, text: str) -> None:
    pdf.set_text_color(20, 40, 80)
    _para(pdf, text, h=7, size=12, style="B")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def _image_row(pdf: _ReportPDF, items: list[tuple[np.ndarray, str]], img_w: float = 56.0) -> None:
    """Place a centered row of images with a small caption under each."""
    if not items:
        return
    pils = [(_arr_to_pil(arr), caption) for arr, caption in items]
    gap = 6.0
    n = len(pils)
    total = n * img_w + (n - 1) * gap
    start_x = _MARGIN + max(0.0, (_USABLE_WIDTH - total) / 2.0)
    y = pdf.get_y()

    # Page-break guard: images do not trigger auto page breaks.
    max_h = max(img_w * pil.size[1] / pil.size[0] for pil, _ in pils)
    if y + max_h + 10 > pdf.h - pdf.b_margin:
        pdf.add_page()
        y = pdf.get_y()

    for i, (pil, caption) in enumerate(pils):
        x = start_x + i * (img_w + gap)
        pdf.image(pil, x=x, y=y, w=img_w)
        pdf.set_xy(x, y + img_w * pil.size[1] / pil.size[0] + 1)
        pdf.set_font("Helvetica", "", 7)
        pdf.multi_cell(img_w, 3.5, _ascii(caption), align="C", new_x="LMARGIN", new_y="NEXT")

    # Reset the cursor below the row (image placement leaves x at the far right).
    pdf.set_xy(_MARGIN, y + max_h + 8)


def _lesion_table(pdf: _ReportPDF, lesions: list[dict[str, Any]]) -> None:
    if not lesions:
        _para(
            pdf,
            "No AI-Identified Suspicious Areas were segmented above the configured "
            "minimum area. Absence of a segmented region does not rule out disease; "
            "all cases Require Professional Review.",
            h=5, size=10,
        )
        pdf.ln(2)
        return

    headers = ["Region", "Label", "Area (px)", "Centroid (x, y)", "BBox (x,y,w,h)"]
    widths = [16.0, 58.0, 22.0, 34.0, 50.0]  # sum = 180
    line_h = 6.0

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(225, 232, 245)
    for text, w in zip(headers, widths):
        pdf.cell(w, line_h, _ascii(text), border=1, align="C", fill=True)
    pdf.ln(line_h)

    pdf.set_font("Helvetica", "", 9)
    for row in _lesion_rows(lesions):
        if pdf.get_y() + line_h > pdf.h - pdf.b_margin:
            pdf.add_page()
        for text, w in zip(row, widths):
            pdf.cell(w, line_h, _ascii(text), border=1)
        pdf.ln(line_h)
    pdf.ln(2)


def _disclaimer_banner(pdf: _ReportPDF, text: str) -> None:
    pdf.set_fill_color(255, 244, 214)
    pdf.set_draw_color(200, 150, 0)
    pdf.set_text_color(120, 80, 0)
    _para(pdf, text, h=5, size=9, style="B", align="C", fill=True, border=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(3)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_report_pdf(
    bundle: dict[str, Any],
    safe_meta: dict[str, Any] | None = None,
    *,
    case_id: str | None = None,
    created_at: str | None = None,
    original_image: np.ndarray | None = None,
    output_path: str | Path | None = None,
    model_statuses: list[ModelStatus] | None = None,
    data_provenance: str | None = None,
) -> bytes:
    """Render the explanation ``bundle`` into a vector PDF and return its bytes.

    ``bundle`` is an ``explain.explain_case`` result. ``safe_meta`` is optional,
    already-PHI-free technical metadata (no identifiers). ``original_image`` (a
    ``[0,1]`` float or uint8 array) is embedded alongside the Grad-CAM and lesion
    overlays when provided.

    ``model_statuses`` and ``data_provenance`` make the report scientifically
    honest and self-contained: the model load state (untrained / loaded / failed)
    and where the input came from (synthetic vs user-provided) are printed in a
    prominent block on page 1 and repeated in the technical metadata, so a reader
    who only has the PDF still knows a demo score has no predictive meaning.

    If ``output_path`` is given the PDF is also written to disk. Raises
    ``ValueError`` if any text would violate the clinical guardrails.
    """
    interp = dict(bundle.get("interpretation", {}) or {})
    lesions = list(bundle.get("lesions", []) or [])
    created_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Guardrail sweep BEFORE we draw anything.
    _assert_report_text_safe(
        _collect_report_text(
            bundle, safe_meta, case_id, created_at, model_statuses, data_provenance
        )
    )

    index_value = interp.get("index_value")
    if index_value is None and "suspicion_index" in bundle:
        index_value = f"{float(bundle['suspicion_index']):.4f}"
    index_name = interp.get("index_name", SUSPICION_INDEX_NAME)
    assessment = interp.get("assessment", "Requires Professional Review")
    recommendation = interp.get("recommendation", "Requires Professional Review")
    disclaimer = str(bundle.get("disclaimer") or DECISION_DISCLAIMER)

    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_title(_ascii(REPORT_TITLE))
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
    pdf.add_page()

    # --- Title -------------------------------------------------------------
    pdf.set_text_color(20, 40, 80)
    _para(pdf, REPORT_TITLE, h=9, size=16, style="B", align="C")
    pdf.set_text_color(0, 0, 0)
    _para(
        pdf,
        f"Case ID: {case_id or '(not persisted)'}    Generated (UTC): {created_at}",
        h=5, size=9, align="C",
    )
    pdf.ln(2)

    # --- Disclaimer banner (top, unmissable) -------------------------------
    _disclaimer_banner(pdf, disclaimer)

    # --- Model status & data provenance (honesty block, page 1) ------------
    if model_statuses is not None or data_provenance is not None:
        statuses = model_statuses or []
        mode = overall_mode(statuses) if statuses else "unknown"
        is_demo = any(s.is_demo for s in statuses) or not statuses
        # A red banner for demonstration output; amber for loaded checkpoints.
        if is_demo:
            pdf.set_fill_color(250, 224, 224)
            pdf.set_draw_color(180, 40, 40)
            pdf.set_text_color(150, 20, 20)
            banner = (
                "DEMONSTRATION OUTPUT - NO PREDICTIVE MEANING. The scores and "
                "regions in this report were produced by untrained model(s) with "
                "random weights (or a simulated fallback) to verify software "
                "behavior only. They are NOT predictions, probabilities of cancer, "
                "risk categories, or clinical assessments."
            )
        else:
            pdf.set_fill_color(255, 244, 214)
            pdf.set_draw_color(200, 150, 0)
            pdf.set_text_color(120, 80, 0)
            banner = (
                "Produced from loaded model checkpoint(s). This application does "
                "NOT verify how those checkpoints were trained or evaluated; "
                "treat performance as unestablished until independently validated."
            )
        _para(pdf, banner, h=5, size=9, style="B", align="C", fill=True, border=1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(0, 0, 0)
        pdf.ln(1)

        _para(pdf, f"Analysis mode: {mode}", h=5, size=9, style="B")
        for status in statuses:
            _para(pdf, f"- {status.short()}", h=4.5, size=9)
        if data_provenance:
            _para(pdf, f"Input data provenance: {data_provenance}", h=5, size=9)
        pdf.ln(3)

    # --- Suspicion summary -------------------------------------------------
    _heading(pdf, "Model Assessment")
    _para(pdf, f"{index_name}: {index_value}", size=11, style="B")
    _para(pdf, f"Assessment: {assessment}", size=11)
    _para(pdf, f"Recommendation: {recommendation}", size=11)
    _para(
        pdf,
        f"Segmented AI-Identified Suspicious Areas: "
        f"{bundle.get('num_suspicious_regions', len(lesions))}",
        size=11,
    )
    pdf.ln(3)

    # --- Visual explanation ------------------------------------------------
    _heading(pdf, "Visual Explanation")
    row: list[tuple[np.ndarray, str]] = []
    if original_image is not None:
        row.append((original_image, "Preprocessed input (PHI-removed)"))
    if bundle.get("gradcam_overlay") is not None:
        row.append((bundle["gradcam_overlay"], "Grad-CAM attention (model focus)"))
    if bundle.get("lesion_overlay") is not None:
        row.append((bundle["lesion_overlay"], "Numbered AI-Identified Suspicious Areas"))
    _image_row(pdf, row)
    pdf.set_text_color(90, 90, 90)
    _para(
        pdf,
        "Heatmap and outlines indicate regions the model attended to / segmented. "
        "They are explanatory aids, not clinical findings.",
        h=4, size=8, style="I",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # --- Lesion table ------------------------------------------------------
    _heading(pdf, "Segmented Regions")
    _lesion_table(pdf, lesions)

    # --- Technical metadata ------------------------------------------------
    if safe_meta or model_statuses or data_provenance:
        _heading(pdf, "Technical Metadata (non-identifying)")
        if model_statuses:
            _para(pdf, f"analysis_mode: {overall_mode(model_statuses)}", h=5, size=9)
            for status in model_statuses:
                _para(pdf, f"model_status: {status.short()}", h=5, size=9)
        if data_provenance:
            _para(pdf, f"data_provenance: {data_provenance}", h=5, size=9)
        for key in sorted(safe_meta or {}):
            _para(pdf, f"{key}: {safe_meta[key]}", h=5, size=9)
        pdf.ln(2)

    out = pdf.output()
    pdf_bytes = bytes(out)
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(pdf_bytes)
    return pdf_bytes
