import builtins
import copy
import json
import re
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import fitz
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
UPLOADS_DIR = ROOT_DIR / "uploads"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from backend.full_pipeline import run_full_pipeline
from backend.feedback_generator import generate_feedback_packages


@contextmanager
def non_interactive_review():
    original_input = builtins.input
    builtins.input = lambda prompt="": ""
    try:
        yield
    finally:
        builtins.input = original_input


def save_uploaded_file(uploaded_file, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as file:
        file.write(uploaded_file.getbuffer())


def slugify_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip()).strip("_") or "run"


def prepare_source_from_upload(uploaded_file, run_root):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".zip":
        source_path = run_root / uploaded_file.name
        save_uploaded_file(uploaded_file, source_path)
        extracted_dir = run_root / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path, "r") as archive:
            archive.extractall(extracted_dir)
        return source_path

    sheets_dir = run_root / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    if suffix in {".png", ".jpg", ".jpeg"}:
        save_uploaded_file(uploaded_file, sheets_dir / uploaded_file.name)
        return sheets_dir

    if suffix == ".pdf":
        pdf_bytes = uploaded_file.getbuffer()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        base_name = Path(uploaded_file.name).stem

        for page_index, page in enumerate(pdf_doc, start=1):
            pix = page.get_pixmap()
            output_path = sheets_dir / f"{base_name}_page_{page_index}.png"
            pix.save(str(output_path))

        pdf_doc.close()
        return sheets_dir

    raise ValueError("Unsupported file type. Upload a PDF, PNG/JPG image, or ZIP file.")


def prepare_source_from_multiple_images(uploaded_files, run_root):
    sheets_dir = run_root / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    saved_any = False
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            continue
        save_uploaded_file(uploaded_file, sheets_dir / uploaded_file.name)
        saved_any = True

    if not saved_any:
        raise ValueError("Upload at least one PNG/JPG image.")

    return sheets_dir


def run_streamlit_pipeline(source_path, answer_key_path, group_by, manifest_path=None):
    with st.status("Running OCR, clustering, and grading...", expanded=True) as status:
        status.write("Running isolated pipeline for this upload")
        with st.spinner("Running OCR, clustering, and grading..."):
            with non_interactive_review():
                result_bundle = run_full_pipeline(
                    source=str(source_path),
                    answer_key_path=answer_key_path,
                    group_by=group_by,
                    manifest_path=str(manifest_path) if manifest_path else None,
                )
        results = result_bundle["results"]
        summary = result_bundle["summary"]
        output_path = result_bundle["output_path"]
        clustered_csv_path = result_bundle["clustered_csv_path"]
        results_json_path = result_bundle["results_json_path"]
        status.update(label="Pipeline complete", state="complete", expanded=False)

    return results, summary, output_path, clustered_csv_path, results_json_path, result_bundle["run_dir"]


def build_reviews_from_results(results):
    reviews = []
    for question_id, clusters in results.items():
        for cluster in clusters:
            semantic = cluster.get("semantic_evaluation", {})
            reviews.append({
                "question_id": question_id,
                "cluster_id": int(cluster.get("cluster_id")),
                "final_marks": semantic.get("suggested_marks_display", "manual review required"),
                "teacher_note": semantic.get("reason", "Cluster review required."),
                "source": "streamlit-reviewed-results",
            })
    return reviews


def run_streamlit_feedback_generation(results, answer_key_path, clustered_csv_path, grading_output_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        review_path = temp_root / "streamlit_reviews.json"
        with open(review_path, "w", encoding="utf-8") as file:
            json.dump({"reviews": build_reviews_from_results(results)}, file, indent=2, ensure_ascii=False)

        backend_run_dir = Path(st.session_state.get("backend_run_dir", BACKEND_DIR))
        feedback_output_dir = backend_run_dir / "exports"
        return generate_feedback_packages(
            review_path=review_path,
            clustered_csv_path=clustered_csv_path,
            answer_key_path=answer_key_path,
            grading_output_path=grading_output_path,
            output_dir=feedback_output_dir,
        )


def apply_cluster_override(results, question_id, cluster_id, marks_value, total_marks, reason_value):
    updated = copy.deepcopy(results)

    for cluster in updated.get(question_id, []):
        if cluster.get("cluster_id") != cluster_id:
            continue

        semantic = cluster.setdefault("semantic_evaluation", {})
        display_value = f"{marks_value}/{total_marks}"
        semantic["suggested_marks"] = marks_value
        semantic["suggested_marks_display"] = display_value
        semantic["reason"] = reason_value
        semantic["manual_reviewed"] = True

        for result in cluster.get("results", []):
            result["suggested_marks"] = display_value
            result["suggested_reason"] = reason_value

        break

    return updated


def build_cluster_overview_df(results):
    rows = []
    for question_id, clusters in results.items():
        for cluster in clusters:
            semantic = cluster.get("semantic_evaluation", {})
            suggested_marks = semantic.get("suggested_marks")
            script_types = sorted(
                {
                    result.get("script_type", "unknown")
                    for result in cluster.get("results", [])
                    if result.get("script_type")
                }
            )
            rows.append({
                "question_id": question_id,
                "cluster_id": cluster.get("cluster_id"),
                "cluster_label": f"{question_id}-C{cluster.get('cluster_id')}",
                "cluster_size": cluster.get("cluster_size", 0),
                "avg_score": cluster.get("avg_score", 0),
                "confidence": semantic.get("confidence", 0),
                "suggested_marks": suggested_marks if suggested_marks is not None else 0,
                "total_marks": semantic.get("total_marks", 0),
                "threshold_passed": semantic.get("passed_similarity_threshold", False),
                "manual_reviewed": semantic.get("manual_reviewed", False),
                "script_types": ", ".join(script_types) if script_types else "unknown",
            })
    return pd.DataFrame(rows)


def get_cluster_by_label(results, cluster_label):
    for question_id, clusters in results.items():
        for cluster in clusters:
            label = f"{question_id}-C{cluster.get('cluster_id')}"
            if label == cluster_label:
                return question_id, cluster
    return None, None


def render_cost_summary(summary):
    st.subheader("Cost And Efficiency")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sheets", summary.get("sheet_count", 0))
    col2.metric("OCR (s)", summary["stage_times"].get("ocr_seconds", 0))
    col3.metric("Clustering (s)", summary["stage_times"].get("clustering_seconds", 0))
    col4.metric("Grading (s)", summary["stage_times"].get("grading_seconds", 0))

    batches = summary.get("batches", [])
    if batches:
        st.dataframe(pd.DataFrame(batches), use_container_width=True)
    st.caption(summary.get("token_estimation_note", ""))


def render_dashboard_graphs(results, summary):
    overview_df = build_cluster_overview_df(results)
    if overview_df.empty:
        st.info("No cluster data available yet.")
        return

    st.subheader("Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Questions", overview_df["question_id"].nunique())
    col2.metric("Total Clusters", len(overview_df))
    col3.metric("Total Answers", int(overview_df["cluster_size"].sum()))
    col4.metric("Avg Cluster Confidence", round(float(overview_df["confidence"].mean()), 2))

    metric_left, metric_right = st.columns(2)
    script_breakdown = (
        overview_df["script_types"]
        .str.split(", ")
        .explode()
        .replace("", "unknown")
        .value_counts()
    )
    reviewed_count = int(overview_df["manual_reviewed"].sum())
    metric_left.metric("Reviewed Clusters", reviewed_count)
    metric_right.metric("Detected Script Types", int(script_breakdown.shape[0]))

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Cluster Size Distribution")
        st.bar_chart(
            overview_df.set_index("cluster_label")["cluster_size"],
            use_container_width=True,
        )

    with right_col:
        st.markdown("### Suggested Marks By Cluster")
        st.bar_chart(
            overview_df.set_index("cluster_label")[["suggested_marks", "total_marks"]],
            use_container_width=True,
        )

    st.markdown("### Threshold Status")
    threshold_counts = (
        overview_df["threshold_passed"]
        .map({True: "Passed", False: "Manual Review"})
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count")
    )
    if not threshold_counts.empty:
        st.bar_chart(threshold_counts.set_index("status")["count"], use_container_width=True)

    if not script_breakdown.empty:
        st.markdown("### Script Distribution")
        st.bar_chart(script_breakdown, use_container_width=True)

    batches = summary.get("batches", [])
    if batches:
        batch_df = pd.DataFrame(batches)
        st.markdown("### Batch Processing Time")
        st.bar_chart(
            batch_df.set_index("batch_id")["estimated_processing_seconds"],
            use_container_width=True,
        )

    st.markdown("### Cluster Overview Table")
    st.dataframe(overview_df, use_container_width=True)


def build_outlier_df(results):
    rows = []
    for question_id, clusters in results.items():
        for cluster in clusters:
            semantic = cluster.get("semantic_evaluation", {})
            cluster_id = cluster.get("cluster_id")
            is_outlier = (
                cluster_id == -1
                or not semantic.get("passed_similarity_threshold", False)
                or semantic.get("confidence", 0) < 0.6
            )
            if not is_outlier:
                continue

            rows.append({
                "question_id": question_id,
                "cluster_id": cluster_id,
                "cluster_label": f"{question_id}-C{cluster_id}",
                "cluster_size": cluster.get("cluster_size", 0),
                "avg_score": cluster.get("avg_score", 0),
                "confidence": semantic.get("confidence", 0),
                "threshold_passed": semantic.get("passed_similarity_threshold", False),
                "suggested_marks_display": semantic.get("suggested_marks_display", "n/a"),
                "reason": semantic.get("reason", ""),
            })
    return pd.DataFrame(rows)


def render_outlier_graphs(results):
    outlier_df = build_outlier_df(results)
    if outlier_df.empty:
        st.success("No outlier clusters detected in the current run.")
        return

    st.subheader("Outlier Clusters")

    col1, col2, col3 = st.columns(3)
    col1.metric("Outlier Clusters", len(outlier_df))
    col2.metric("Outlier Answers", int(outlier_df["cluster_size"].sum()))
    col3.metric(
        "Avg Outlier Confidence",
        round(float(outlier_df["confidence"].mean()), 2),
    )

    st.markdown("### Outlier Cluster Sizes")
    st.bar_chart(
        outlier_df.set_index("cluster_label")["cluster_size"],
        use_container_width=True,
    )

    st.markdown("### Outlier Confidence")
    st.bar_chart(
        outlier_df.set_index("cluster_label")["confidence"],
        use_container_width=True,
    )

    st.markdown("### Outlier Summary")
    st.dataframe(
        outlier_df[
            [
                "question_id",
                "cluster_id",
                "cluster_size",
                "avg_score",
                "confidence",
                "threshold_passed",
                "suggested_marks_display",
                "reason",
            ]
        ],
        use_container_width=True,
    )


def render_downloads():
    results = st.session_state.get("results")
    summary = st.session_state.get("summary")
    clustered_csv_path = st.session_state.get("clustered_csv_path")
    feedback_packages = st.session_state.get("feedback_packages")

    if results is None:
        return

    st.subheader("Downloads")
    st.download_button(
        "Download reviewed_output.json",
        data=json.dumps(results, indent=2),
        file_name="reviewed_output.json",
        mime="application/json",
    )

    if summary is not None:
        st.download_button(
            "Download cost_efficiency_log.json",
            data=json.dumps(summary, indent=2),
            file_name="cost_efficiency_log.json",
            mime="application/json",
        )

    if clustered_csv_path and Path(clustered_csv_path).exists():
        with open(clustered_csv_path, "r", encoding="utf-8") as file:
            st.download_button(
                "Download final_clustered_grades.csv",
                data=file.read(),
                file_name="final_clustered_grades.csv",
                mime="text/csv",
            )

    if feedback_packages:
        feedback_path = Path(feedback_packages["feedback_path"])
        if feedback_path.exists():
            with open(feedback_path, "r", encoding="utf-8") as file:
                st.download_button(
                    "Download student_feedback.json",
                    data=file.read(),
                    file_name="student_feedback.json",
                    mime="application/json",
                )


def load_cluster_answer_texts(clustered_csv_path, question_id, cluster_id):
    clustered_csv_path = Path(clustered_csv_path)
    if not clustered_csv_path.exists():
        return []

    df = pd.read_csv(clustered_csv_path)
    if {"question_id", "cluster_id", "answer_text"}.issubset(df.columns):
        filtered = df[(df["question_id"] == question_id) & (df["cluster_id"] == cluster_id)]
        rows = []
        for _, row in filtered.iterrows():
            rows.append({
                "student_id": str(row.get("student_id", "")),
                "answer_text": str(row.get("answer_text", "")).strip(),
            })
        return rows

    cluster_column = f"{question_id}_Cluster_ID"
    answer_column = f"{question_id}_Answer"
    if cluster_column not in df.columns or answer_column not in df.columns:
        return []

    filtered = df[df[cluster_column] == cluster_id]
    rows = []
    for _, row in filtered.iterrows():
        rows.append({
            "student_id": str(row.get("student_id", "")),
            "answer_text": str(row.get(answer_column, "")).strip(),
        })
    return rows


def load_cluster_image_paths(results_json_path, student_ids, saved_dataset_path=None):
    results_json_path = Path(results_json_path) if results_json_path else None
    saved_dataset_path = Path(saved_dataset_path) if saved_dataset_path else None

    if not results_json_path or not results_json_path.exists():
        return []

    with open(results_json_path, "r", encoding="utf-8") as file:
        ocr_rows = json.load(file)

    student_ids = {str(student_id) for student_id in student_ids}
    image_rows = []

    for row in ocr_rows:
        student_id = str(row.get("student_id", ""))
        if student_id not in student_ids:
            continue

        for source_file in row.get("source_files", []):
            source_path = Path(source_file)
            resolved_path = None

            if source_path.exists():
                resolved_path = source_path
            elif saved_dataset_path:
                candidate_paths = [
                    saved_dataset_path / source_path,
                    saved_dataset_path / "sheets" / source_path.name,
                    saved_dataset_path / source_path.name,
                ]
                for candidate in candidate_paths:
                    if candidate.exists():
                        resolved_path = candidate
                        break
                if resolved_path is None:
                    matches = list(saved_dataset_path.rglob(source_path.name))
                    if matches:
                        resolved_path = matches[0]
            if resolved_path is None:
                global_matches = list(UPLOADS_DIR.rglob(source_path.name))
                if global_matches:
                    resolved_path = global_matches[0]

            image_rows.append({
                "student_id": student_id,
                "image_path": str(resolved_path) if resolved_path else None,
                "source_file": str(source_file),
            })

    return image_rows


def load_cluster_rows(results_json_path, student_ids):
    results_json_path = Path(results_json_path) if results_json_path else None
    if not results_json_path or not results_json_path.exists():
        return []

    with open(results_json_path, "r", encoding="utf-8") as file:
        ocr_rows = json.load(file)

    wanted = {str(student_id) for student_id in student_ids}
    return [
        row for row in ocr_rows
        if str(row.get("student_id", "")) in wanted
    ]


def render_feedback_packages(feedback_packages):
    if not feedback_packages:
        st.info("No student feedback packages have been generated yet.")
        return

    feedback_rows = feedback_packages.get("student_feedback", [])
    st.subheader("Student Tutoring Feedback")
    st.caption(f"Generated {len(feedback_rows)} feedback packages.")

    if not feedback_rows:
        return

    feedback_df = pd.DataFrame(feedback_rows)
    st.dataframe(
        feedback_df[
            [
                "student_id",
                "question_id",
                "cluster_id",
                "final_marks",
                "used_model",
                "pdf_path",
                "email_path",
            ]
        ],
        use_container_width=True,
    )

    preview_student = st.selectbox(
        "Preview student feedback",
        feedback_df["student_id"].astype(str).tolist(),
        key="feedback_preview_student",
    )
    selected = next(
        row for row in feedback_rows
        if str(row.get("student_id")) == str(preview_student)
    )
    with st.container(border=True):
        st.markdown(f"**Student ID:** {selected['student_id']}")
        st.markdown(f"**Question:** {selected['question_id']}")
        st.markdown(f"**Marks:** {selected['final_marks']}")
        st.markdown("**Tutoring Paragraph**")
        st.write(selected["tutoring_paragraph"])
        st.markdown("**Practice Question**")
        st.write(selected["practice_question"])
        st.markdown("**Email Text**")
        st.code(selected["email_text"], language="text")
