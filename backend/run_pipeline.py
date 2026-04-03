from csv_loader import load_csv, load_teacher_answers
from cluster_processor import group_by_cluster
from grading.scoring_engine import grade_cluster
from rubric_generator import generate_rubric
from difflib import get_close_matches
import json
import os

# Load student data
data = load_csv("final_clustered_grades.csv")

# Load teacher answers
teacher_answers = load_teacher_answers("Model_Answer_Momentum.csv")

# Group clusters
clusters = group_by_cluster(data)

final_output = []

# All teacher questions
teacher_questions = list(teacher_answers.keys())

for cluster_id, answers in clusters.items():

    # Get first question
    question = list(teacher_answers.keys())[0]

    teacher_data = teacher_answers[question]

    rubric = generate_rubric(teacher_data)

    print("RUBRIC:", rubric)

    cluster_payload = {
        "cluster_id": cluster_id,
        "answers": answers,
        "rubric": rubric
    }

    result = grade_cluster(cluster_payload)
    final_output.append(result)

# Save output
output_path = os.path.join(os.path.dirname(__file__), "output.json")

with open(output_path, "w") as f:
    json.dump(final_output, f, indent=2)

print("✅ Output file updated at:", output_path)
