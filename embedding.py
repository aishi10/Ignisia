import json
import re
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from csv_loader import load_teacher_answers
from text_normalizer import build_embedding_text, clean_ocr_artifacts


def _question_marker_patterns(number):
    return [
        rf"\bQ\.?\s*{number}\b",
        rf"\bQue\s*{number}\b",
        rf"\bQuestion\s*{number}\b",
        rf"\bAns?:?\s*{number}\b",
        rf"\bAnswer\s*{number}\b",
        rf"प्रश्न\s*[:.]?\s*{number}\b",
        rf"उत्तर\s*[:.]?\s*{number}\b",
        rf"\b{number}[\.\):>\]]",
    ]


def split_answers_by_question(text, question_ids):
    normalized_text = clean_ocr_artifacts(text)
    if not question_ids:
        return {}
    if len(question_ids) == 1:
        return {question_ids[0]: normalized_text}

    positions = []
    for index, question_id in enumerate(question_ids, start=1):
        start_match = None
        for pattern in _question_marker_patterns(index):
            start_match = re.search(pattern, normalized_text, re.IGNORECASE)
            if start_match:
                break
        positions.append((question_id, start_match.start() if start_match else None))

    if all(position is None for _, position in positions):
        answers = {question_ids[0]: normalized_text}
        for question_id in question_ids[1:]:
            answers[question_id] = "No Answer"
        return answers

    resolved_positions = []
    last_known = 0
    for question_id, position in positions:
        if position is None:
            position = last_known
        else:
            last_known = position
        resolved_positions.append((question_id, position))

    answers = {}
    for idx, (question_id, start_pos) in enumerate(resolved_positions):
        end_pos = len(normalized_text)
        for _, next_pos in resolved_positions[idx + 1 :]:
            if next_pos > start_pos:
                end_pos = next_pos
                break
        chunk = normalized_text[start_pos:end_pos].strip() if start_pos is not None else ""
        answers[question_id] = chunk or "No Answer"

    return answers


def _normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _reassign_outliers(cluster_ids, embeddings, similarity_threshold=0.58):
    cluster_ids = np.array(cluster_ids, dtype=int)
    normalized_embeddings = _normalize_embeddings(np.asarray(embeddings))
    cluster_centroids = {}
    for cluster_id in sorted(set(cluster_ids.tolist())):
        if cluster_id == -1:
            continue
        vectors = normalized_embeddings[cluster_ids == cluster_id]
        if len(vectors) == 0:
            continue
        centroid = vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            continue
        cluster_centroids[cluster_id] = centroid / centroid_norm

    if not cluster_centroids:
        return cluster_ids

    for index, cluster_id in enumerate(cluster_ids):
        if cluster_id != -1:
            continue
        similarities = {
            candidate: float(np.dot(normalized_embeddings[index], centroid))
            for candidate, centroid in cluster_centroids.items()
        }
        if not similarities:
            continue
        best_cluster, best_similarity = max(similarities.items(), key=lambda item: item[1])
        if best_similarity >= similarity_threshold:
            cluster_ids[index] = best_cluster
    return cluster_ids


def cluster_answers(
    results_json_path,
    output_csv_path,
    output_json_path,
    answer_key_path=None,
    include_flagged=True,
):
    results_json_path = Path(results_json_path)
    output_csv_path = Path(output_csv_path)
    output_json_path = Path(output_json_path)
    backend_dir = Path(__file__).resolve().parent
    answer_key_path = Path(answer_key_path or backend_dir / "Answer_Key_Q1_Q2.csv")

    print("--- Loading JSON OCR Data ---")
    with open(results_json_path, "r", encoding="utf-8") as file:
        ocr_data = json.load(file)

    if not ocr_data:
        raise ValueError("OCR results are empty. Cannot cluster answers.")

    teacher_answers = load_teacher_answers(answer_key_path)
    question_ids = [question_id for question_id in teacher_answers.keys() if str(question_id).strip()]

    rows = []
    for row in ocr_data:
        if not include_flagged and row.get("flagged"):
            continue

        split_answers = split_answers_by_question(row.get("full_text", ""), question_ids)
        for question_id in question_ids:
            if not str(question_id).strip():
                continue
            answer_text = split_answers.get(question_id, "No Answer")
            rows.append({
                "student_id": str(row.get("student_id", "")),
                "document_id": row.get("document_id", row.get("student_id", "")),
                "question_id": question_id,
                "answer_text": answer_text,
                "embedding_text": build_embedding_text(answer_text),
                "script_type": row.get("script_type", "unknown"),
                "avg_confidence": row.get("avg_confidence", 0),
                "flagged": row.get("flagged", False),
                "grouping_strategy": row.get("grouping_strategy", ""),
                "source_files": json.dumps(row.get("source_files", []), ensure_ascii=False),
            })

    answers_df = pd.DataFrame(rows)
    if answers_df.empty:
        raise ValueError("No answers were extracted for clustering.")

    print("--- Loading the SentenceTransformer Model ---")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_epsilon=0.05,
    )

    clustered_frames = []
    for question_id, question_df in answers_df.groupby("question_id", sort=False):
        if not str(question_id).strip():
            continue
        print(f"\n--- CLUSTERING {question_id} ---")
        texts = question_df["embedding_text"].fillna("").tolist()
        if len(question_df) == 1:
            cluster_ids = np.array([0], dtype=int)
        else:
            embeddings = model.encode(texts)
            cluster_ids = clusterer.fit_predict(embeddings)
            cluster_ids = _reassign_outliers(cluster_ids, embeddings)

        clustered_question_df = question_df.copy()
        clustered_question_df["cluster_id"] = cluster_ids
        clustered_frames.append(clustered_question_df)

    export_df = pd.concat(clustered_frames, ignore_index=True)

    print("\n=== FINAL CLUSTERING RESULTS ===")
    print(export_df[["student_id", "question_id", "cluster_id"]].to_string(index=False))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_csv_path, index=False)
    export_df.to_json(output_json_path, orient="records", indent=2, force_ascii=False)

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
        answer_key_path=backend_dir / "Answer_Key_Q1_Q2.csv",
    )
