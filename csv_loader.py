def load_csv(filepath):
    import csv

    data = []

    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:

            student_id = row.get("student_id")

            # Q1
            data.append({
                "student_id": str(student_id),
                "cluster_id": int(row.get("Q1_Cluster_ID", 0)),
                "question_id": "Q1",
                "raw_text": row.get("Q1_Answer", "")
            })

            # Q2
            data.append({
                "student_id": str(student_id),
                "cluster_id": int(row.get("Q2_Cluster_ID", 0)),
                "question_id": "Q2",
                "raw_text": row.get("Q2_Answer", "")
            })

    return data


def load_teacher_answers(filepath):
    import csv

    rubric_map = {}

    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            qid = row["Question_ID"]

            rubric_map[qid] = {
                "question": row.get("Question", ""),
                "model_answer": row["Model_Answer"],
                "required": row["Required_Elements"]
            }

    return rubric_map
