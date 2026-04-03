import json
import re
from pathlib import Path

import hdbscan
import pandas as pd
from sentence_transformers import SentenceTransformer


def extract_q1(text):
    match = re.search(
        r"(?:Q\.? ?1|Que 1|Ans?:? 1).*?(?=Q\.? ?2|Que 2|Ans?:? 2|Any: 2|9\.2|2>|2\)|$)",
        str(text),
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(0).strip()

    fallback_text = re.sub(
        r"(?:Q\.? ?2|Que 2|Ans?:? 2|Any: 2|9\.2|2>|2\)).*",
        "",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fallback_text.strip()


def extract_q2(text):
    match = re.search(
        r"(?:Q\.? ?2|Que 2|Ans?:? ?2|Any: ?2|9\.?2\)?|2>|2\)|0?2\]|92\]).*",
        str(text),
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        cleaned_text = re.sub(
            r"(Teacher's Signature|AMAR KRISH)",
            "",
            match.group(0),
            flags=re.IGNORECASE,
        )
        return cleaned_text.strip()
    return "No Answer"


def cluster_answers(
    results_json_path,
    output_csv_path,
    output_json_path,
    include_flagged=True,
):
    results_json_path = Path(results_json_path)
    output_csv_path = Path(output_csv_path)
    output_json_path = Path(output_json_path)

    print("--- Loading JSON OCR Data ---")
    with open(results_json_path, "r", encoding="utf-8") as file:
        ocr_data = json.load(file)

    df = pd.DataFrame(ocr_data)
    if df.empty:
        raise ValueError("OCR results are empty. Cannot cluster answers.")

    clean_df = df.copy() if include_flagged else df[df["flagged"] == False].copy()

    clean_df["Q1_Answer"] = clean_df["full_text"].apply(extract_q1)
    clean_df["Q2_Answer"] = clean_df["full_text"].apply(extract_q2)

    print("--- Loading the SentenceTransformer Model ---")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_epsilon=0.05,
    )

    print("\n--- PASS 1: Clustering Question 1 ---")
    q1_embeddings = model.encode(clean_df["Q1_Answer"].tolist())
    clean_df["Q1_Cluster_ID"] = clusterer.fit_predict(q1_embeddings)

    print("--- PASS 2: Clustering Question 2 ---")
    q2_embeddings = model.encode(clean_df["Q2_Answer"].tolist())
    clean_df["Q2_Cluster_ID"] = clusterer.fit_predict(q2_embeddings)

    print("\n=== FINAL CLUSTERING RESULTS ===")
    print(clean_df[["student_id", "Q1_Cluster_ID", "Q2_Cluster_ID"]])

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = clean_df.drop(columns=["full_text"], errors="ignore")
    export_df.to_csv(output_csv_path, index=False)
    export_df.to_json(output_json_path, orient="records", indent=2)

    print(f"Saved clustered CSV: {output_csv_path}")
    print(f"Saved clustered JSON: {output_json_path}")

    return export_df


if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    cluster_answers(
        results_json_path=project_root / "ocr_output" / "results.json",
        output_csv_path=backend_dir / "final_clustered_grades.csv",
        output_json_path=backend_dir / "final_clustered_grades.json",
    )
