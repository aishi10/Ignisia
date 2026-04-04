import csv
from pathlib import Path


def load_csv(filepath):
    filepath = Path(filepath)
    data = []

    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        is_long_format = {"student_id", "question_id", "answer_text", "cluster_id"}.issubset(fieldnames)

        for row in reader:
            student_id = str(row.get("student_id", ""))

            if is_long_format:
                question_id = str(row.get("question_id", "")).strip()
                if not question_id:
                    continue
                data.append({
                    "student_id": student_id,
                    "document_id": row.get("document_id", student_id),
                    "question_id": question_id,
                    "cluster_id": int(row.get("cluster_id", -1) or -1),
                    "raw_text": row.get("answer_text", ""),
                    "embedding_text": row.get("embedding_text", ""),
                    "script_type": row.get("script_type", "unknown"),
                    "source_files": row.get("source_files", ""),
                })
                continue

            for field in fieldnames:
                if not field.endswith("_Answer"):
                    continue
                question_id = field[:-7]
                if not question_id.strip():
                    continue
                cluster_field = f"{question_id}_Cluster_ID"
                data.append({
                    "student_id": student_id,
                    "document_id": row.get("document_id", student_id),
                    "question_id": question_id,
                    "cluster_id": int(row.get(cluster_field, -1) or -1),
                    "raw_text": row.get(field, ""),
                    "embedding_text": row.get(f"{question_id}_Embedding_Text", ""),
                    "script_type": row.get("script_type", "unknown"),
                    "source_files": row.get("source_files", ""),
                })

    return data


def load_teacher_answers(filepath):
    filepath = Path(filepath)
    rubric_map = {}

    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            qid = row["Question_ID"]
            rubric_map[qid] = {
                "question_id": qid,
                "question": row.get("Question", ""),
                "model_answer": row.get("Model_Answer", ""),
                "required": row.get("Required_Elements", ""),
                "key_concepts": row.get("Key_Concepts", ""),
            }

    return rubric_map
