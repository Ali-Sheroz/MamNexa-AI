"""MamNexa AI - Streamlit research dashboard (Phases III-IV).

A local, single-page decision-*support* demo. It preprocesses an uploaded
mammogram (PHI removed), runs the Phase-II explainability bundle (Model
Malignancy Suspicion Index + Grad-CAM attention + numbered AI-Identified
Suspicious Areas), renders a vector PDF pre-analysis report, and can persist the
analysis to Supabase (or an in-memory ephemeral store) with a strict
**Right to Erasure** hard-purge. A Phase-IV molecular panel shows TCGA-BRCA
pathway context (cell-cycle regulation, DNA damage response, ...) alongside the
imaging heatmaps.

Clinical + scientific-integrity guardrails: this interface never confirms or
rules out cancer, and the molecular panel is a SEPARATE, non-patient-matched
TCGA-BRCA reference cohort -- never the imaged patient's biology. Every result is
framed as research decision-support that Requires Professional Review.

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import model as m
from src import molecular as mol
from src import segmentation as seg
from src.backend import SupabaseBackend
from src.config import (
    DECISION_DISCLAIMER,
    IMAGE_SIZE,
    LICENSE_NOTICE,
    MODELS_DIR,
    MOLECULAR_DISCLAIMER,
    REPORT_TITLE,
    SUSPICION_INDEX_NAME,
)
from src.explain import explain_case
from src.preprocessing import array_to_uint8, dicom_to_clean_array
from src.report import build_report_pdf, np_to_png_bytes

CLASSIFIER_PATH = MODELS_DIR / "classifier.keras"
UNET_PATH = MODELS_DIR / "unet.keras"


# ---------------------------------------------------------------------------
# Cached resources (models + backend live once per process, in memory)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models() -> tuple[Any, Any, list[str]]:
    """Load exported weights if present, else build structural (untrained) models.

    Returns ``(classifier, segmenter, untrained)`` where ``untrained`` names the
    models that fell back to random weights so the UI can flag it honestly.
    """
    untrained: list[str] = []
    if CLASSIFIER_PATH.exists():
        classifier = m.load_trained_model(CLASSIFIER_PATH)
    else:
        classifier, _ = m.build_efficientnet_b0(weights=None)
        m.compile_model(classifier)
        untrained.append("classifier")

    if UNET_PATH.exists():
        segmenter = seg.load_trained_unet(UNET_PATH)
    else:
        segmenter = seg.build_unet()
        seg.compile_unet(segmenter)
        untrained.append("segmenter")
    return classifier, segmenter, untrained


@st.cache_resource(show_spinner=False)
def get_backend() -> SupabaseBackend:
    return SupabaseBackend()


@st.cache_data(show_spinner=False)
def demo_cohort() -> pd.DataFrame:
    """Deterministic synthetic TCGA-BRCA reference cohort for the offline demo.

    This is SYNTHETIC structure for wiring/verification -- not real TCGA-BRCA
    expression and not this patient's tissue (see the molecular disclaimer).
    """
    return mol.synthetic_expression_matrix()


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
    return rgb, {"source_format": suffix.lstrip("."), "ingest": "non-DICOM demo upload"}


def build_analysis_metadata(bundle: dict[str, Any], safe_meta: dict[str, Any], untrained: list[str]) -> dict[str, Any]:
    """Assemble the non-identifying row payload for storage."""
    interp = bundle.get("interpretation", {})
    meta = dict(safe_meta)
    meta.update(
        {
            "suspicion_index": round(float(bundle["suspicion_index"]), 4),
            "assessment": interp.get("assessment", "Requires Professional Review"),
            "num_suspicious_regions": int(bundle.get("num_suspicious_regions", 0)),
            "models_untrained": ",".join(untrained) if untrained else "none",
        }
    )
    return meta


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="MamNexa AI", page_icon="🔬", layout="wide")

    classifier, segmenter, untrained = load_models()
    backend = get_backend()

    # -- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.header("MamNexa AI")
        st.caption("Explainable AI research framework - breast imaging")
        mode = "Ephemeral (in-memory)" if backend.ephemeral else "Supabase (PostgreSQL + S3)"
        st.write(f"**Storage backend:** {mode}")
        if backend.ephemeral:
            st.caption(
                "No SUPABASE_URL / SUPABASE_KEY detected - analyses are kept only "
                "in memory for this session. See .env.example to enable Supabase."
            )
        st.divider()
        st.caption(LICENSE_NOTICE)

    st.title("Research Decision Support")
    st.warning(DECISION_DISCLAIMER, icon="⚠️")
    if untrained:
        st.info(
            "Model(s) running on **random (untrained) weights**: "
            f"{', '.join(untrained)}. Outputs verify the pipeline mechanics only "
            "and are **not clinically meaningful** until trained on CBIS-DDSM.",
            icon="🧪",
        )

    uploaded = st.file_uploader(
        "Upload a mammogram (DICOM .dcm preferred; PNG/JPG accepted for demo)",
        type=["dcm", "dicom", "png", "jpg", "jpeg"],
    )

    if uploaded is not None and st.button("Run analysis", type="primary"):
        _run_and_store_analysis(uploaded, classifier, segmenter, untrained, backend)

    _render_results()
    _render_molecular()
    _render_erasure(backend)


def _run_and_store_analysis(uploaded, classifier, segmenter, untrained, backend) -> None:
    with st.spinner("Preprocessing and analyzing..."):
        try:
            image, safe_meta = load_uploaded_image(uploaded)
            bundle = explain_case(image, classifier, segmenter)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.error(f"Analysis failed: {exc}")
            return

        base_uint8 = array_to_uint8(image)
        metadata = build_analysis_metadata(bundle, safe_meta, untrained)
        pdf_bytes = build_report_pdf(
            bundle, safe_meta=safe_meta, original_image=base_uint8
        )
        image_png = np_to_png_bytes(base_uint8)

    st.session_state["bundle"] = bundle
    st.session_state["base_uint8"] = base_uint8
    st.session_state["safe_meta"] = safe_meta
    st.session_state["pdf_bytes"] = pdf_bytes
    st.session_state["image_png"] = image_png
    st.session_state["metadata"] = metadata
    # Reset any prior stored id (this is a fresh case).
    st.session_state.pop("analysis_id", None)


def _render_results() -> None:
    bundle = st.session_state.get("bundle")
    if not bundle:
        return

    interp = bundle["interpretation"]
    st.subheader("Model Assessment")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(SUSPICION_INDEX_NAME, interp["index_value"])
    with c2:
        st.write(f"**Assessment:** {interp['assessment']}")
        st.write(f"**Recommendation:** {interp['recommendation']}")
        st.write(f"**Segmented AI-Identified Suspicious Areas:** {bundle['num_suspicious_regions']}")

    st.subheader("Visual Explanation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state["base_uint8"], caption="Preprocessed input (PHI-removed)", channels="RGB")
    with col2:
        st.image(bundle["gradcam_overlay"], caption="Grad-CAM attention (model focus)", channels="RGB")
    with col3:
        st.image(bundle["lesion_overlay"], caption="Numbered AI-Identified Suspicious Areas", channels="RGB")
    st.caption(
        "Heatmaps and outlines are explanatory aids showing where the model "
        "attended / segmented - not clinical findings."
    )

    if bundle["lesions"]:
        st.subheader("Segmented Regions")
        st.table(
            [
                {
                    "Region": f"#{l['region_id']}",
                    "Label": l["label"],
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
    st.subheader("🧬 Molecular Pathway Context (TCGA-BRCA) - Research Extension")
    st.warning(MOLECULAR_DISCLAIMER, icon="⚠️")

    # -- Choose the reference cohort ---------------------------------------
    source = st.radio(
        "Reference expression cohort",
        ["Built-in synthetic TCGA-BRCA demo cohort", "Upload expression matrix (CSV / TSV)"],
        help=(
            "The molecular panel scores biological pathways relative to a reference "
            "cohort. The built-in cohort is deterministic SYNTHETIC data for offline "
            "demonstration - not real patient expression."
        ),
    )

    expr: pd.DataFrame | None = None
    if source.startswith("Upload"):
        expr_file = st.file_uploader(
            "Expression matrix - first column = gene symbol, remaining columns = samples",
            type=["csv", "tsv", "txt"],
            key="expr_upload",
        )
        if expr_file is not None:
            try:
                expr = _load_expression_upload(expr_file)
            except Exception as exc:  # noqa: BLE001 - surface parse failures to the user
                st.error(f"Could not read expression matrix: {exc}")
                return
        else:
            st.info("Upload a genes x samples matrix, or switch to the built-in demo cohort.")
            return
    else:
        expr = demo_cohort()

    if expr is None or expr.shape[1] == 0:
        st.info("No samples available in the selected cohort.")
        return

    st.caption(
        f"Reference cohort: **{expr.shape[1]} samples x {expr.shape[0]} genes**. "
        "Pathway scores below are cohort-relative (per-gene z-scored across these samples)."
    )

    sample = st.selectbox("Reference sample to profile", list(expr.columns))
    try:
        bundle = mol.analyze_sample(expr, sample=sample)
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
            "Run an imaging analysis above to display the Model Malignancy Suspicion "
            "Index alongside this molecular context (the two remain separate, "
            "non-patient-matched cohorts)."
        )


def _render_erasure(backend: SupabaseBackend) -> None:
    st.divider()
    with st.expander("🗑️ Right to Erasure - permanently delete a stored analysis"):
        st.caption(
            "Executes a hard purge: the metadata row AND the stored image/PDF "
            "objects are deleted, then re-checked to verify nothing remains."
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
                if st.session_state.get("analysis_id") == target.strip():
                    st.session_state.pop("analysis_id", None)
            else:
                st.error(
                    "Purge did not fully verify. "
                    f"Row deleted: {result.row_deleted}; verified gone: {result.verified_gone}. "
                    "The ID may not exist, or the backend rejected the delete."
                )


if __name__ == "__main__":
    main()
