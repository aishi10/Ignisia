from pathlib import Path

import streamlit as st

from app_utils import (
    UPLOADS_DIR,
    prepare_source_from_multiple_images,
    prepare_source_from_upload,
    run_streamlit_pipeline,
    save_uploaded_file,
    slugify_name,
)


st.set_page_config(page_title="Ignisia Home", page_icon="📝", layout="wide")

st.title("Ignisia Grading Dashboard")
st.write("Upload student answer sheets and an answer-key CSV to run OCR, clustering, grading, and cost reporting.")

with st.sidebar:
    st.header("Run Setup")
    group_by = st.selectbox("Grouping mode", ["auto", "folder", "filename", "single"], index=0)

answer_key_file = st.file_uploader("Answer key CSV", type=["csv"])
manifest_file = st.file_uploader("Manifest (optional)", type=["json", "csv"])
sheet_input_file = st.file_uploader(
    "Answer sheets input (single PDF, single image, or ZIP)",
    type=["pdf", "png", "jpg", "jpeg", "zip"],
)
sheet_image_files = st.file_uploader(
    "Or upload multiple answer-sheet images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)

if run_clicked:
    if answer_key_file is None:
        st.error("Upload the answer-key CSV before running.")
    elif sheet_input_file is None and not sheet_image_files:
        st.error("Upload a PDF, PNG/JPG image, ZIP file, or multiple answer-sheet images.")
    else:
        input_name = (
            Path(sheet_input_file.name).stem
            if sheet_input_file is not None
            else f"image_batch_{len(sheet_image_files)}"
        )
        run_label = slugify_name(input_name)
        run_root = UPLOADS_DIR / run_label
        if run_root.exists():
            suffix_counter = 1
            while (UPLOADS_DIR / f"{run_label}_{suffix_counter}").exists():
                suffix_counter += 1
            run_root = UPLOADS_DIR / f"{run_label}_{suffix_counter}"

        run_root.mkdir(parents=True, exist_ok=True)
        answer_key_path = run_root / answer_key_file.name
        save_uploaded_file(answer_key_file, answer_key_path)

        manifest_path = None
        if manifest_file is not None:
            manifest_path = run_root / manifest_file.name
            save_uploaded_file(manifest_file, manifest_path)

        if sheet_input_file is not None:
            source_path = prepare_source_from_upload(sheet_input_file, run_root)
        else:
            source_path = prepare_source_from_multiple_images(sheet_image_files, run_root)

        results, summary, output_path, clustered_csv_path, results_json_path, backend_run_dir = run_streamlit_pipeline(
            source_path=source_path,
            answer_key_path=answer_key_path,
            group_by=group_by,
            manifest_path=manifest_path,
        )

        st.session_state["results"] = results
        st.session_state["summary"] = summary
        st.session_state["output_path"] = str(output_path)
        st.session_state["clustered_csv_path"] = str(clustered_csv_path)
        st.session_state["results_json_path"] = str(results_json_path)
        st.session_state["saved_dataset_path"] = str(run_root)
        st.session_state["answer_key_path"] = str(answer_key_path)
        st.session_state["backend_run_dir"] = str(backend_run_dir)
        st.session_state["feedback_packages"] = None

        st.success("Pipeline run completed.")
        st.info(f"Saved uploaded dataset at: {run_root}")
        st.caption(f"Isolated backend run directory: {backend_run_dir}")
        st.page_link("pages/1_Dashboard.py", label="Open Dashboard", icon="📊")
        st.page_link("pages/2_Override_Review.py", label="Open Override Review", icon="🛠️")

if st.session_state.get("saved_dataset_path"):
    st.caption(f"Latest saved dataset: {st.session_state['saved_dataset_path']}")
