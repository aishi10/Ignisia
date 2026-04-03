def load_csv(filepath):
    import csv

    data = []

    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            data.append({
                "student_id": str(row["ID"]),
                "cluster_id": int(row["Cluster_ID"]),
                "question": row.get("Question", "Q1"),  # 🔥 IMPORTANT
                "raw_text": row["Student Answer"]
            })

    return data
def load_teacher_answers(filepath):
    import csv

    rubric_map = {}

    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            question = row.get("Question", "Q1")

            model_answer = row.get("Model Answer", "").strip()
            required = row.get("Required Elements", "").strip()

            rubric_map[question] = {
                "model_answer": model_answer,
                "required": required
            }

    return rubric_map
