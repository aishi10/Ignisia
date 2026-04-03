from grading.regex_parser import extract_math_expressions
from grading.math_validator import validate_equation
from grading.keyword_matcher import keyword_score
from collections import Counter
import re
import time

def highlight_keywords(text, keywords):
    for kw in keywords:
        text = re.sub(f"(?i)({kw})", r"<mark>\1</mark>", text)
    return text


def grade_cluster(cluster_payload):
    start_time = time.time()

    results = []
    rubric = cluster_payload["rubric"]

    for ans in cluster_payload["answers"]:
        student_text = ans["raw_text"].strip()

        # Normalize multiline OCR
        student_text = student_text.replace("\n", " ")

        # 🚨 Empty / garbage detection
        if not student_text or not re.search(r'[a-zA-Z]', student_text):
            results.append({
                "student_id": ans["student_id"],
                "score": 0,
                "math_correct": False,
                "keyword_score": 0,
                "confidence": 0,
                "partial_credit_math_error": False,
                "highlighted_text": student_text,
                "feedback": ["Invalid or empty answer"]
            })
            continue

        # Keyword scoring
        kw_score = keyword_score(student_text, rubric["keywords"])
        optional_score = keyword_score(
            student_text,
            rubric.get("optional_keywords", [])
        )

        # Highlight text for frontend
        highlighted_text = highlight_keywords(student_text, rubric["keywords"])

        # Extract math
        student_math = extract_math_expressions(student_text)

        # Support multiple correct equations
        correct_equations = rubric.get("equation")
        if isinstance(correct_equations, str):
            correct_equations = [correct_equations]

        formula_match = False
        math_correct = False
        partial_credit_math_error = False

        if correct_equations and student_math:
            formula_match = any(
                validate_equation(expr, eq)
                for eq in correct_equations
                for expr in student_math
            )

            math_correct = formula_match  # currently symbolic match

            # 🔥 Step 3.3 HARD FLAGGING
            if formula_match and not math_correct:
                partial_credit_math_error = True

        # Final score
        final_score = min(1.0, (
            kw_score * rubric["keyword_weight"] +
            optional_score * 0.2 +
            (1 if math_correct else 0) * rubric["math_weight"]
        ))

        # Feedback
        feedback = set()

        missing_keywords = [
            kw for kw in rubric["keywords"]
            if kw.lower() not in student_text.lower()
        ]

        if missing_keywords:
            feedback.add(f"Missing keywords: {', '.join(missing_keywords)}")

        if math_correct and kw_score < 0.5:
            feedback.add("Correct formula but missing explanation")

        if correct_equations and not math_correct:
            feedback.add("Incorrect or missing formula")

        if partial_credit_math_error:
            feedback.add("Correct formula but calculation error")

        if not feedback:
            feedback.add("Perfect answer")

        # Confidence
        confidence = round((kw_score + (1 if math_correct else 0)) / 2, 2)

        results.append({
            "student_id": ans["student_id"],
            "score": round(final_score, 2),
            "math_correct": math_correct,
            "keyword_score": round(kw_score, 2),
            "confidence": confidence,
            "partial_credit_math_error": partial_credit_math_error,
            "highlighted_text": highlighted_text,
            "feedback": list(feedback)
        })

    # Safe analytics
    if not results:
        return {
            "cluster_id": cluster_payload["cluster_id"],
            "cluster_size": 0,
            "avg_score": 0,
            "top_issues": [],
            "processing_time": 0,
            "results": []
        }

    avg_score = sum(r["score"] for r in results) / len(results)

    # Aggregate feedback
    all_feedback = []
    for r in results:
        all_feedback.extend(r["feedback"])

    top_issues = [
        issue for issue, _ in Counter(all_feedback).most_common(3)
    ]

    processing_time = round(time.time() - start_time, 2)

    return {
        "cluster_id": cluster_payload["cluster_id"],
        "cluster_size": len(results),
        "avg_score": round(avg_score, 2),
        "top_issues": top_issues,
        "processing_time": processing_time,
        "results": results
    }
