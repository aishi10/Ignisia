import streamlit as st
import pandas as pd

from app_utils import (
    apply_cluster_override,
    load_cluster_image_paths,
    render_downloads,
    render_feedback_packages,
    run_streamlit_feedback_generation,
)


def find_pending_outlier(results):
    for question_id, clusters in results.items():
        for cluster in clusters:
            semantic = cluster.get("semantic_evaluation", {})
            if cluster.get("cluster_id") == -1 and not semantic.get("manual_reviewed", False):
                return question_id, cluster
    return None, None


def render_cluster_override_form(question_id, cluster, blocking=False):
    semantic = cluster.get("semantic_evaluation", {})
    cluster_id = cluster.get("cluster_id")
    title = f"Cluster {cluster_id} • {cluster.get('cluster_size', 0)} answers • {semantic.get('suggested_marks_display', 'n/a')}"

    if blocking:
        st.markdown(f"### {title}")
        st.write("This outlier cluster must be reviewed before continuing.")
    else:
        st.markdown(f"### {title}")

    st.write("Reason:", semantic.get("reason", ""))
    st.write("Confidence:", semantic.get("confidence", ""))
    st.write("Threshold passed:", semantic.get("passed_similarity_threshold", False))
    st.write("Top issues:", ", ".join(cluster.get("top_issues", [])) or "None")

    student_ids = [result.get("student_id") for result in cluster.get("results", [])]
    results_json_path = st.session_state.get("results_json_path")
    saved_dataset_path = st.session_state.get("saved_dataset_path")
    image_rows = load_cluster_image_paths(
        results_json_path=results_json_path,
        student_ids=student_ids,
        saved_dataset_path=saved_dataset_path,
    )

    if image_rows:
        st.write("Student Answer Script Images")
        for answer in image_rows:
            with st.container(border=True):
                st.markdown(f"**Student ID:** {answer['student_id']}")
                if answer["image_path"]:
                    st.image(answer["image_path"], use_container_width=True)
                else:
                    st.write(f"Image not found for source: {answer['source_file']}")

    similarity_rows = semantic.get("variation_similarity_scores", [])
    if similarity_rows:
        st.write("Variation Similarities")
        st.dataframe(pd.DataFrame(similarity_rows), use_container_width=True)

    total_marks = float(semantic.get("total_marks", 0) or 0)
    current_marks = semantic.get("suggested_marks")
    default_marks = float(current_marks) if current_marks is not None else 0.0
    reason_default = semantic.get("reason", "")

    override_marks = st.number_input(
        "Override marks",
        min_value=0.0,
        max_value=float(total_marks) if total_marks > 0 else 100.0,
        value=default_marks,
        step=0.5,
        key=f"marks_{question_id}_{cluster_id}_{'blocking' if blocking else 'normal'}",
    )
    override_reason = st.text_area(
        "Override reason",
        value=reason_default,
        key=f"reason_{question_id}_{cluster_id}_{'blocking' if blocking else 'normal'}",
        height=100,
    )

    submitted = st.button(
        "Save override",
        key=f"apply_{question_id}_{cluster_id}_{'blocking' if blocking else 'normal'}",
        type="primary" if blocking else "secondary",
    )

    if submitted:
        st.session_state["results"] = apply_cluster_override(
            st.session_state["results"],
            question_id=question_id,
            cluster_id=cluster_id,
            marks_value=round(float(override_marks), 2),
            total_marks=semantic.get("total_marks", 0),
            reason_value=override_reason,
        )
        st.success(f"Applied override for {question_id} cluster {cluster_id}.")
        st.rerun()


st.set_page_config(page_title="Ignisia Review", page_icon="🛠️", layout="wide")
st.title("Override Review")

results = st.session_state.get("results")

if not results:
    st.info("Run the pipeline from the home page first to review clusters.")
else:
    if st.session_state.get("saved_dataset_path"):
        st.caption(f"Saved dataset: {st.session_state['saved_dataset_path']}")

    pending_question_id, pending_cluster = find_pending_outlier(results)

    if pending_cluster is not None:
        if hasattr(st, "dialog"):
            @st.dialog("Outlier Cluster Review")
            def outlier_review_dialog():
                render_cluster_override_form(pending_question_id, pending_cluster, blocking=True)

            outlier_review_dialog()
            st.info("Complete the outlier override in the dialog before continuing with the rest of the review.")
        else:
            st.subheader("Outlier Review Required")
            with st.container(border=True):
                render_cluster_override_form(pending_question_id, pending_cluster, blocking=True)
        st.stop()

    for question_id, clusters in results.items():
        if not str(question_id).strip():
            continue
        st.markdown(f"## {question_id}")
        if not clusters:
            st.info(f"No clusters found for {question_id}.")
            continue

        for cluster in clusters:
            semantic = cluster.get("semantic_evaluation", {})
            cluster_id = cluster.get("cluster_id")
            title = (
                f"Cluster {cluster_id} • "
                f"{cluster.get('cluster_size', 0)} answers • "
                f"{semantic.get('suggested_marks_display', 'n/a')}"
            )

            with st.expander(title):
                render_cluster_override_form(question_id, cluster, blocking=False)

                result_rows = cluster.get("results", [])
                if result_rows:
                    st.write("Student Results")
                    st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

    st.divider()
    st.subheader("Tutoring Feedback Packages")
    st.write("Generate personalized tutoring feedback, practice questions, and export files from the reviewed cluster results.")

    if st.button("Generate Student Feedback Packages", type="primary", use_container_width=True):
        with st.spinner("Generating tutoring feedback packages..."):
            feedback_packages = run_streamlit_feedback_generation(
                results=st.session_state["results"],
                answer_key_path=st.session_state.get("answer_key_path"),
                clustered_csv_path=st.session_state.get("clustered_csv_path"),
                grading_output_path=st.session_state.get("output_path"),
            )
            st.session_state["feedback_packages"] = feedback_packages
        st.success("Student feedback packages generated.")

    render_feedback_packages(st.session_state.get("feedback_packages"))

    render_downloads()
