import json
from pathlib import Path

from cluster_processor import group_by_cluster
from csv_loader import load_csv, load_teacher_answers
from grading.scoring_engine import grade_cluster
from rubric_generator import generate_rubric


def run_grading_pipeline(
    clustered_csv_path,
    answer_key_path=None,
    output_path=None,
):
    backend_dir = Path(__file__).resolve().parent
    clustered_csv_path = Path(clustered_csv_path)
    answer_key_path = Path(answer_key_path or backend_dir / "Answer_Key_Q1_Q2.csv")
    output_path = Path(output_path or backend_dir / "output.json")

    data = load_csv(clustered_csv_path)
    teacher_answers = load_teacher_answers(answer_key_path)
    clusters = group_by_cluster(data)

    final_output = {
        question_id: []
        for question_id in teacher_answers.keys()
        if str(question_id).strip()
    }

    for (qid, cluster_id), answers in clusters.items():
        if not str(qid).strip():
            continue
        teacher_data = teacher_answers.get(qid, {})
        rubric = generate_rubric(teacher_data)

        print(f"\nCLUSTER: {cluster_id} | QUESTION: {qid}")
        print(f"RUBRIC: {rubric}")

        cluster_payload = {
            "cluster_id": cluster_id,
            "question_id": qid,
            "answers": answers,
            "rubric": rubric,
        }

        result = grade_cluster(cluster_payload)
        final_output.setdefault(qid, []).append(result)

    for qid in final_output:
        final_output[qid].sort(key=lambda item: item["cluster_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(final_output, file, indent=2)

    print(f"\nOutput saved at: {output_path}")
    return final_output


if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parent
    run_grading_pipeline(
        clustered_csv_path=backend_dir / "final_clustered_grades.csv",
        answer_key_path=backend_dir / "Answer_Key_Q1_Q2.csv",
        output_path=backend_dir / "output.json",
    )
