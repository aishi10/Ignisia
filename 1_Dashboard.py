import streamlit as st

from app_utils import (
    build_cluster_overview_df,
    get_cluster_by_label,
    load_cluster_rows,
    load_cluster_image_paths,
    render_cost_summary,
    render_dashboard_graphs,
)


st.set_page_config(page_title="Ignisia Dashboard", page_icon="📊", layout="wide")
st.title("Dashboard")

results = st.session_state.get("results")
summary = st.session_state.get("summary")

if not results or not summary:
    st.info("Run the pipeline from the home page first to see dashboard graphs.")
else:
    render_cost_summary(summary)
    render_dashboard_graphs(results, summary)

    overview_df = build_cluster_overview_df(results)
    if not overview_df.empty:
        st.markdown("### Cluster Drilldown")
        selected_cluster_label = st.selectbox(
            "Select a cluster to view answer sheets",
            overview_df["cluster_label"].tolist(),
        )

        selected_question_id, selected_cluster = get_cluster_by_label(results, selected_cluster_label)
        if selected_cluster is not None:
            semantic = selected_cluster.get("semantic_evaluation", {})
            st.write(
                f"Showing answer sheets for {selected_cluster_label} "
                f"({selected_cluster.get('cluster_size', 0)} answers)"
            )
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            detail_col1.metric("Suggested Marks", semantic.get("suggested_marks_display", "n/a"))
            detail_col2.metric("Confidence", semantic.get("confidence", 0))
            detail_col3.metric("Manual Review", "Yes" if semantic.get("manual_reviewed") else "No")
            st.caption(semantic.get("reason", ""))

            student_ids = [result.get("student_id") for result in selected_cluster.get("results", [])]
            ocr_rows = load_cluster_rows(
                results_json_path=st.session_state.get("results_json_path"),
                student_ids=student_ids,
            )
            if ocr_rows:
                st.markdown("**Cluster Script Mix**")
                script_types = sorted({row.get("script_type", "unknown") for row in ocr_rows})
                st.write(", ".join(script_types))

            image_rows = load_cluster_image_paths(
                results_json_path=st.session_state.get("results_json_path"),
                student_ids=student_ids,
                saved_dataset_path=st.session_state.get("saved_dataset_path"),
            )

            if image_rows:
                for answer in image_rows:
                    with st.container(border=True):
                        st.markdown(f"**Student ID:** {answer['student_id']}")
                        if answer["image_path"]:
                            st.image(answer["image_path"], use_container_width=True)
                        else:
                            st.write(f"Image not found for source: {answer['source_file']}")
            else:
                st.info("No answer-sheet images found for this cluster.")
