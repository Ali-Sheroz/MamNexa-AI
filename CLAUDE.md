# Project: MamNexa AI
An open-source, non-commercial explainable artificial intelligence framework for mammographic and molecular analysis of breast cancer[cite: 1]. 

## 1. Core Directives & Identity
*   **Academic Purpose:** This project is strictly for research portfolio evaluation to support graduate admissions in Medical Biotechnology. It is licensed under Creative Commons NonCommercial (CC BY-NC 4.0) and must not include any monetization hooks.
*   **Clinical Guardrails:** The system must be framed as research decision support, not as a tool that independently confirms or rules out cancer[cite: 1]. 
*   **Mandatory Vocabulary:** Avoid outputs such as "confirmed cancer," "definitely damaged tissue," or "safe tissue"[cite: 1]. All interface and PDF outputs must use cautious phrasing, such as "AI-Identified Suspicious Area", "Model Malignancy Suspicion Index", and "Requires Professional Review".
*   **Scientific Integrity:** Never allow images from the same patient to leak across partitions[cite: 1]. Patient-level dataset splitting is mandatory.

## 2. Technology Stack
*   **Core Logic:** Google Colab, Python, TensorFlow/Keras, pydicom, OpenCV, and NumPy[cite: 1].
*   **Backend & Database:** Supabase (PostgreSQL for metadata + S3 buckets for temporary image/PDF storage) and the supabase-py SDK.
*   **Interface & Reporting:** Streamlit for the demo interface[cite: 1] and fpdf2 for vector PDF report generation.

## 3. Execution Phases
Claude Code must execute development in strict, isolated modules:
*   **Phase I (Imaging Baseline):** CBIS-DDSM dataset preparation, metadata stripping, patient-level splits, and EfficientNet-B0 baseline classifier[cite: 1].
*   **Phase II (Localization & Explainability):** U-Net segmentation, OpenCV contour extraction/numbering, and Grad-CAM attention heatmaps[cite: 1].
*   **Phase III (Interface & Erasure):** Streamlit dashboard, Supabase backend integration, PDF pre-analysis report generation, and immediate data-deletion logic.
*   **Phase IV (Molecular Extension):** TCGA-BRCA transcriptomic analysis mapped to biological pathways (e.g., cell-cycle control, DNA repair)[cite: 1]. Do not initiate this phase until Phases I-III are locked.

## 4. Operational Rules for Claude Code
*   **Local Inference:** The architectures themselves are not paid products[cite: 1]. Models must run locally in memory using @st.cache_resource after loading exported weight files.
*   **Right to Erasure:** Implement a strict deletion flow. When a user requests data deletion, execute a hard purge of the PostgreSQL metadata row and the corresponding Supabase S3 file.
*   **Verification Check:** After writing a major script (like DICOM preprocessing), provide a local test script to verify it functions correctly before moving to the next architectural phase.