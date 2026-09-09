"""MamNexa AI - Streamlit research dashboard (Phases III-IV).

A local, single-page decision-*support* demo. It preprocesses an uploaded
mammogram (PHI removed), runs the Phase-II explainability bundle (Model
Malignancy Suspicion Index + Grad-CAM attention + numbered AI-Identified
Suspicious Areas), renders a vector PDF pre-analysis report, and can persist the
analysis to Supabase (or an in-memory ephemeral store) with a strict
**Right to Erasure** hard-purge. A Phase-IV molecular panel shows cohort-relative
pathway context (cell-cycle regulation, DNA damage response, ...) alongside the
imaging heatmaps.

Honesty first: the app ships without trained weights. When a checkpoint is
absent the model runs on random weights and EVERY resulting output is labeled a
"Demonstration output - no predictive meaning". A checkpoint that is present but
fails to load is surfaced as an error, never silently replaced with random
weights. The classifier and segmenter statuses are reported separately.

Clinical + scientific-integrity guardrails: this interface never confirms or
rules out cancer, and the molecular panel is a SEPARATE, non-patient-matched
reference cohort -- never the imaged patient's biology. Every result Requires
Professional Review.

Run:  streamlit run app.py   (Windows: use port 8600, see .claude/launch.json)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import model as m
from src import model_status as ms
from src import molecular as mol
from src import segmentation as seg
from src.backend import SupabaseBackend
from src.config import (
    CLASSIFIER_CHECKPOINT,
    DECISION_DISCLAIMER,
    IMAGE_SIZE,
    LICENSE_NOTICE,
    MOLECULAR_DISCLAIMER,
    SEGMENTER_CHECKPOINT,
    SUSPICION_INDEX_NAME,
)
from src.explain import explain_case
from src.preprocessing import array_to_uint8, dicom_to_clean_array
from src.report import build_report_pdf, np_to_png_bytes

CLASSIFIER_NAME = "classifier (EfficientNet-B0)"
SEGMENTER_NAME = "segmenter (U-Net)"


# ---------------------------------------------------------------------------
# Cached resources (models + models' load status live once per process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models() -> tuple[Any, Any, list[ms.ModelStatus]]:
    """Load exported checkpoints if present, else build untrained demo models.

    Returns ``(classifier, segmenter, statuses)``. A missing checkpoint falls
    back to random weights (a labeled demonstration). A checkpoint that EXISTS
    but fails to load is reported as ``load_failed`` -- we never silently
    substitute random weights for a broken checkpoint, because that would hide a
    real problem behind a plausible-looking demo.
    """
    classifier, clf_status = _load_one(
        CLASSIFIER_CHECKPOINT, CLASSIFIER_NAME,
        loader=m.load_trained_model,
        builder=lambda: _built(m.build_efficientnet_b0(weights=None)[0], m.compile_model),
        load_error=m.CheckpointLoadError,
    )
    segmenter, seg_status = _load_one(
        SEGMENTER_CHECKPOINT, SEGMENTER_NAME,
        loader=seg.load_trained_unet,
        builder=lambda: _built(seg.build_unet(), seg.compile_unet),
        load_error=seg.CheckpointLoadError,
    )
    return classifier, segmenter, [clf_status, seg_status]


def _built(model: Any, compile_fn: Any) -> Any:
    compile_fn(model)
    return model


def _load_one(checkpoint: Path, name: str, *, loader, builder, load_error) -> tuple[Any, ms.ModelStatus]:
    """Load one checkpoint with honest status; build untrained model only if absent."""
    if not Path(checkpoint).exists():
        return builder(), ms.ModelStatus(
            name=name, state=ms.STATE_UNTRAINED, checkpoint=str(checkpoint),
            detail="No checkpoint file found; using random (untrained) weights.",
        )
    try:
        model = loader(checkpoint)
    except load_error as exc:
        # Present but broken: build a demo model to keep the UI alive, but the
        # status is load_failed so the error is shown, NOT masked as a demo.
        return builder(), ms.ModelStatus(
            name=name, state=ms.STATE_LOAD_FAILED, checkpoint=str(checkpoint),
            detail=str(exc),
        )
    return model, ms.ModelStatus(
        name=name, state=ms.STATE_LOADED, checkpoint=str(checkpoint),
        detail="Checkpoint loaded; training provenance/quality NOT verified by the app.",
    )


@st.cache_resource(show_spinner=False)
def _shared_backend() -> SupabaseBackend:
    """The process-wide backend. In Supabase mode this is genuinely shared.

    For the EPHEMERAL fallback we do NOT reuse this instance across browser
    sessions (see :func:`get_backend`) -- an in-memory store shared across users
    would leak one visitor's analyses to another.
    """
    return SupabaseBackend()


def get_backend() -> SupabaseBackend:
    """Return a backend that is session-isolated in ephemeral mode.

    Supabase-backed storage is inherently shared (that is the point), so the
    cached client is reused. The in-memory ephemeral store, however, is created
    once PER SESSION and kept in ``st.session_state`` so one browser session can
    never read or erase another's data.
    """
    shared = _shared_backend()
    if not shared.ephemeral:
        return shared
    if "ephemeral_backend" not in st.session_state:
        st.session_state["ephemeral_backend"] = SupabaseBackend()
    return st.session_state["ephemeral_backend"]


@st.cache_data(show_spinner=False)
def demo_cohort() -> pd.DataFrame:
    """Deterministic synthetic reference cohort for the offline molecular demo.

    This is SYNTHETIC structure for wiring/verification -- not real TCGA-BRCA
    expression and not this patient's tissue (see the molecular disclaimer).
    """
    return mol.synthetic_expression_matrix()


def demo_mammogram() -> tuple[np.ndarray, dict[str, Any]]:
    """A clearly-labeled synthetic mammogram-like image for the guided demo.

    Deterministic, no dataset needed: a soft breast-shaped intensity gradient
    with a couple of brighter blobs so the pipeline has structure to attend to
    and segment. This is SYNTHETIC test input, not a real mammogram.
    """
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float32)
    # Breast-like radial falloff from an off-center origin.
    cx, cy = IMAGE_SIZE * 0.35, IMAGE_SIZE * 0.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    base = np.clip(1.0 - r / (IMAGE_SIZE * 0.75), 0.0, 1.0) ** 1.5
    # A couple of brighter "density" blobs.
    for (bx, by, br, amp) in [(120, 90, 18, 0.5), (150, 150, 12, 0.4)]:
        base += amp * np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * br ** 2)))
    base += rng.normal(0, 0.02, size=base.shape)  # mild texture
    gray = np.clip(base, 0.0, 1.0).astype(np.float32)
    rgb = np.stack([gray, gray, gray], axis=-1)
    meta = {
        "source_format": "synthetic-demo",
        "ingest": "built-in synthetic mammogram (no dataset required)",
        "synthetic_example": True,
        "possible_burned_in_annotation": False,
    }
    return rgb, meta


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def load_uploaded_image(uploaded: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess an upload into an (H, W, 3) float [0,1] image + safe metadata.

    DICOM goes through the full Phase-I PHI-stripping pipeline. A plain image is
    a demo convenience (grayscale -> resize -> [0,1] -> 3 channels); it carries
    no header identifiers to strip.
    """
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in {".dcm", ".dicom", ""}:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            clean = dicom_to_clean_array(tmp_path, size=IMAGE_SIZE)
        finally:
            # Local scratch file is removed immediately after preprocessing.
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
        return clean.array, dict(clean.safe_meta)

    from PIL import Image

    img = Image.open(uploaded).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rgb = np.stack([arr, arr, arr], axis=-1)
    return rgb, {
        "source_format": suffix.lstrip("."),
        "ingest": "non-DICOM demo upload",
        # A plain image has no DICOM header to check; burned-in text cannot be
        # ruled out from pixels here, so flag it as unknown -> review.
        "possible_burned_in_annotation": True,
    }


def build_analysis_metadata(
    bundle: dict[str, Any],
    safe_meta: dict[str, Any],
    statuses: list[ms.ModelStatus],
    data_provenance: str,
) -> dict[str, Any]:
    """Assemble the non-identifying row payload for storage (provenance preserved)."""
    interp = bundle.get("interpretation", {})
    meta = dict(safe_meta)
    meta.update(
        {
            "suspicion_index": round(float(bundle["suspicion_index"]), 4),
            "assessment": interp.get("assessment", "Requires Professional Review"),
            "num_suspicious_regions": int(bundle.get("num_suspicious_regions", 0)),
            "analysis_mode": ms.overall_mode(statuses),
            "model_status": "; ".join(s.short() for s in statuses),
            "data_provenance": data_provenance,
        }
    )
    return meta


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="MamNexa AI", page_icon="🔬", layout="wide")

    classifier, segmenter, statuses = load_models()
    backend = get_backend()

    _render_sidebar(backend, statuses)

    st.title("Research Decision Support")
    st.warning(DECISION_DISCLAIMER, icon="⚠️")
    _render_model_status_banner(statuses)

    st.subheader("1. Choose an input")
    st.caption(
        "New here? Use **Load synthetic demo example** for a guided, no-download "
        "run. Outputs are clearly labeled as demonstration only."
    )
    dcol, ucol = st.columns([1, 2])
    with dcol:
        if st.button("Load synthetic demo example", type="secondary"):
            image, safe_meta = demo_mammogram()
            _run_and_store_analysis(
                image, safe_meta, classifier, segmenter, statuses, backend,
                data_provenance="Synthetic demo mammogram (built-in; not a real image)",
            )
    with ucol:
        uploaded = st.file_uploader(
            "Or upload a mammogram (DICOM .dcm preferred; PNG/JPG/WebP accepted for demo)",
            type=["dcm", "dicom", "png", "jpg", "jpeg", "webp"],
        )

    if uploaded is not None:
        _handle_upload(uploaded, classifier, segmenter, statuses, backend)

    _render_results(statuses)
    _render_molecular()
    _render_erasure(backend)


def _render_sidebar(backend: SupabaseBackend, statuses: list[ms.ModelStatus]) -> None:
    with st.sidebar:
        st.header("MamNexa AI")
        st.caption("Explainable AI research framework - breast imaging")

        st.markdown("**Model checkpoints**")
        for status in statuses:
            icon = {
                ms.STATE_LOADED: "✅",
                ms.STATE_UNTRAINED: "🧪",
                ms.STATE_LOAD_FAILED: "❌",
            }.get(status.state, "•")
            st.caption(f"{icon} {status.short()}")

        st.divider()
        mode = "Ephemeral (in-memory)" if backend.ephemeral else "Supabase (PostgreSQL + S3)"
        st.write(f"**Storage backend:** {mode}")
        if backend.ephemeral:
            st.caption(
                "No SUPABASE_URL / SUPABASE_KEY detected - analyses are kept only "
                "in memory for THIS browser session and vanish when it ends. See "
                ".env.example to enable Supabase. Live Supabase behavior cannot be "
                "tested without credentials."
            )
        st.divider()
        st.caption(LICENSE_NOTICE)


def _render_model_status_banner(statuses: list[ms.ModelStatus]) -> None:
    failed = [s for s in statuses if s.state == ms.STATE_LOAD_FAILED]
    demo = [s for s in statuses if s.state == ms.STATE_UNTRAINED]
    if failed:
        st.error(
            "**Checkpoint failed to load** for: "
            + ", ".join(s.name for s in failed)
            + ". The file exists but could not be read (corrupt or incompatible "
            "TensorFlow/Keras version). It has NOT been silently replaced with "
            "random weights. Details: " + " | ".join(s.detail for s in failed),
            icon="❌",
        )
    if demo:
        st.info(
            "**Demonstration mode.** Running on random (untrained) weights for: "
            f"{', '.join(s.name for s in demo)}. Every score and region below is a "
            "**Demonstration output - no predictive meaning**; it verifies the "
            "software pipeline only and is not clinically meaningful until trained "
            "on a real dataset (e.g. CBIS-DDSM).",
            icon="🧪",
        )


def _handle_upload(uploaded, classifier, segmenter, statuses, backend) -> None:
    """Preprocess the upload, gate on burned-in PHI, then offer Run analysis."""
    try:
        image, safe_meta = load_uploaded_image(uploaded)
    except Exception as exc:  # noqa: BLE001 - surface preprocessing failure
        st.error(f"Could not read the uploaded file: {exc}")
        return

    flagged = bool(safe_meta.get("possible_burned_in_annotation", False))
    provenance = f"User-provided upload ({uploaded.name})"
    if flagged:
        st.warning(
            "**Possible burned-in annotation.** Removing the DICOM header does NOT "
            "remove text burned into the image pixels. This image is flagged (or its "
            "annotation status is unknown), so it may still contain identifying "
            "text. Confirm it has been reviewed before continuing.",
            icon="⚠️",
        )
        reviewed = st.checkbox(
            "I have reviewed this image and confirm it contains no burned-in "
            "patient information."
        )
        if not reviewed:
            st.stop()

    if st.button("Run analysis", type="primary"):
        _run_and_store_analysis(
            image, safe_meta, classifier, segmenter, statuses, backend,
            data_provenance=provenance,
        )


def _run_and_store_analysis(
    image, safe_meta, classifier, segmenter, statuses, backend, *, data_provenance
) -> None:
    with st.spinner("Preprocessing and analyzing..."):
        try:
            bundle = explain_case(image, classifier, segmenter)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.error(f"Analysis failed: {exc}")
            return

        base_uint8 = array_to_uint8(image)
        metadata = build_analysis_metadata(bundle, safe_meta, statuses, data_provenance)
        pdf_bytes = build_report_pdf(
            bundle, safe_meta=safe_meta, original_image=base_uint8,
            model_statuses=statuses, data_provenance=data_provenance,
        )
        image_png = np_to_png_bytes(base_uint8)

    st.session_state["bundle"] = bundle
    st.session_state["base_uint8"] = base_uint8
    st.session_state["safe_meta"] = safe_meta
    st.session_state["pdf_bytes"] = pdf_bytes
    st.session_state["image_png"] = image_png
    st.session_state["metadata"] = metadata
    st.session_state["data_provenance"] = data_provenance
    # Reset any prior stored id (this is a fresh case).
    st.session_state.pop("analysis_id", None)


def _render_results(statuses: list[ms.ModelStatus]) -> None:
    bundle = st.session_state.get("bundle")
    if not bundle:
        return

    is_demo = any(s.is_demo for s in statuses)
    score_label = "Demo score (no predictive meaning)" if is_demo else SUSPICION_INDEX_NAME
    region_label = "Illustrative region" if is_demo else "AI-Identified Suspicious Area"

    interp = bundle["interpretation"]
    st.subheader("2. Model Assessment")
    if is_demo:
        st.caption(ms.DEMO_OUTPUT_BANNER)
    else:
        st.caption(ms.LOADED_CHECKPOINT_BANNER)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(score_label, interp["index_value"])
    with c2:
        st.write(f"**Assessment:** {interp['assessment']}")
        st.write(f"**Recommendation:** {interp['recommendation']}")
        st.write(
            f"**Segmented {region_label}s:** {bundle['num_suspicious_regions']}"
        )
        st.caption(f"Input provenance: {st.session_state.get('data_provenance', 'unknown')}")

    st.subheader("Visual Explanation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state["base_uint8"], caption="Preprocessed input (PHI-removed)", channels="RGB")
    with col2:
        cap = "Illustrative attention map (demo)" if is_demo else "Grad-CAM attention (model focus)"
        st.image(bundle["gradcam_overlay"], caption=cap, channels="RGB")
    with col3:
        cap = f"Numbered {region_label}s"
        st.image(bundle["lesion_overlay"], caption=cap, channels="RGB")
    st.caption(
        "Heatmaps and outlines are explanatory aids showing where the model "
        "attended / segmented - not clinical findings."
        + (" With untrained weights they reflect random initialization, not learned features."
           if is_demo else "")
    )

    if bundle["lesions"]:
        st.subheader("Segmented Regions")
        st.table(
            [
                {
                    "Region": f"#{l['region_id']}",
                    "Label": (region_label if is_demo else l["label"]),
                    "Area (px)": l["area_px"],
                    "Centroid (x, y)": f"({l['centroid'][0]:.0f}, {l['centroid'][1]:.0f})",
                    "BBox (x, y, w, h)": ", ".join(str(v) for v in l["bbox"]),
                }
                for l in bundle["lesions"]
            ]
        )

    st.subheader("Report & Storage")
    dcol, scol = st.columns(2)
    with dcol:
        st.download_button(
            "Download PDF report",
            data=st.session_state["pdf_bytes"],
            file_name="mamnexa_pre_analysis_report.pdf",
            mime="application/pdf",
        )
    with scol:
        if st.button("Store analysis to backend (temporary)"):
            backend = get_backend()
            stored = backend.store_analysis(
                metadata=st.session_state["metadata"],
                image_png=st.session_state["image_png"],
                pdf_bytes=st.session_state["pdf_bytes"],
            )
            st.session_state["analysis_id"] = stored.analysis_id
            st.success(f"Stored. Analysis ID: {stored.analysis_id}")


def _load_expression_upload(uploaded: Any) -> pd.DataFrame:
    """Read an uploaded expression matrix (CSV or TSV) into a genes x samples frame."""
    sep = "\t" if Path(uploaded.name).suffix.lower() in {".tsv", ".txt"} else ","
    return mol.load_expression_matrix(uploaded, sep=sep)


def _render_molecular() -> None:
    st.divider()
    st.subheader("🧬 Molecular Pathway Context - Research Extension")
    st.warning(MOLECULAR_DISCLAIMER, icon="⚠️")
    st.caption(
        "Method: **mean cohort-relative gene z-score** per pathway (each gene is "
        "standardized across the reference cohort; a pathway's score is the mean of "
        "its genes' z-scores). This is a transparent descriptive statistic - NOT "
        "ssGSEA and NOT a validated measurement of pathway activation."
    )

    # -- Choose the reference cohort ---------------------------------------
    source = st.radio(
        "Reference expression cohort",
        ["Built-in synthetic demo cohort", "Upload expression matrix (CSV / TSV)"],
        help=(
            "The built-in cohort is deterministic SYNTHETIC data for offline "
            "demonstration - not real patient expression. Uploaded matrices are "
            "treated as user-provided and are never relabeled as verified TCGA-BRCA."
        ),
    )

    expr: pd.DataFrame | None = None
    provenance = mol.PROVENANCE_SYNTHETIC
    if source.startswith("Upload"):
        provenance = mol.PROVENANCE_USER
        expr_file = st.file_uploader(
            "Expression matrix - first column = gene symbol, remaining columns = samples "
            f"(>= {mol.MIN_COHORT_SAMPLES} samples required)",
            type=["csv", "tsv", "txt"],
            key="expr_upload",
        )
        if expr_file is not None:
            try:
                expr = _load_expression_upload(expr_file)
            except Exception as exc:  # noqa: BLE001 - surface parse failures
                st.error(f"Could not read expression matrix: {exc}")
                return
        else:
            st.info("Upload a genes x samples matrix, or switch to the built-in demo cohort.")
            return
    else:
        expr = demo_cohort()

    # -- Validate before scoring -------------------------------------------
    try:
        report = mol.validate_cohort(expr)
    except ValueError as exc:
        st.error(f"Cohort cannot be scored: {exc}")
        return

    st.caption(
        f"Reference cohort [{provenance}]: **{report['n_samples']} samples x "
        f"{report['n_genes']} genes**; curated-gene coverage "
        f"{report['curated_genes_measured']}/{report['curated_genes_total']}."
    )
    for note in report["notes"]:
        st.caption(f"⚠️ {note}")

    sample = st.selectbox("Reference sample to profile", list(expr.columns))
    try:
        bundle = mol.analyze_sample(expr, sample=sample, provenance=provenance)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Molecular analysis failed: {exc}")
        return

    scores = bundle["pathway_scores"]
    rows = [s.as_row() for s in scores]

    # -- Pathway activity bar chart (cohort-relative mean-z) ---------------
    chart_df = pd.DataFrame(
        {"Activity score (mean z)": [s.score for s in scores]},
        index=[s.name for s in scores],
    )
    st.bar_chart(chart_df)
    st.caption(
        "Bars show cohort-relative activity (mean gene z-score) per pathway - a "
        "hypothesis-generating research signal, not a diagnosis."
    )

    st.table(rows)
    st.caption(bundle["summary"])

    # -- Side-by-side framing with the imaging result (NOT patient-matched) --
    imaging_bundle = st.session_state.get("bundle")
    if imaging_bundle:
        link = mol.link_imaging_to_molecular(
            float(imaging_bundle["suspicion_index"]), bundle
        )
        st.markdown("**Imaging + molecular context (shown side by side, not combined):**")
        st.info(link["narrative"], icon="🔗")
    else:
        st.caption(
            "Run an imaging analysis above to display the imaging suspicion index "
            "alongside this molecular context (the two remain separate, "
            "non-patient-matched cohorts)."
        )


def _render_erasure(backend: SupabaseBackend) -> None:
    st.divider()
    with st.expander("🗑️ Right to Erasure - permanently delete a stored analysis"):
        st.caption(
            "Executes a hard purge: the metadata row AND the stored image/PDF "
            "objects are deleted, then re-checked to verify nothing remains. This "
            "cannot remove copies you have already downloaded (e.g. the PDF report)."
        )
        default_id = st.session_state.get("analysis_id", "")
        target = st.text_input("Analysis ID to erase", value=default_id)
        confirm = st.checkbox("I understand this is permanent and irreversible.")
        if st.button("Permanently erase", disabled=not (target and confirm)):
            result = backend.purge_analysis(target.strip())
            if result.fully_erased:
                st.success(
                    f"Erased analysis {result.analysis_id}. "
                    f"Row deleted: {result.row_deleted}; objects deleted: "
                    f"{len(result.objects_deleted)}; verified gone: {result.verified_gone}."
                )
                # Clear this session's displayed results + buffers when the erased
                # id matches what we are showing, so nothing lingers in the UI.
                if st.session_state.get("analysis_id") == target.strip():
                    for key in (
                        "analysis_id", "bundle", "base_uint8", "safe_meta",
                        "pdf_bytes", "image_png", "metadata", "data_provenance",
                    ):
                        st.session_state.pop(key, None)
                    st.info("Displayed results and in-memory buffers for this case were cleared.")
            elif result.row_deleted or result.objects_deleted:
                st.warning(
                    "Partial erasure. "
                    f"Row deleted: {result.row_deleted}; objects deleted: "
                    f"{len(result.objects_deleted)}; verified gone: {result.verified_gone}. "
                    "Some artifacts may remain; re-run erasure or check the backend."
                )
            else:
                st.error(
                    "Nothing was erased. "
                    f"Row deleted: {result.row_deleted}; verified gone: {result.verified_gone}. "
                    "The ID may not exist, or the backend rejected the delete."
                )


if __name__ == "__main__":
    main()
