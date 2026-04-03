from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

import re
import time

from sentence_transformers import SentenceTransformer, util

from grading.math_validator import validate_equation
from grading.regex_parser import extract_math_expressions


@lru_cache(maxsize=1)
def get_semantic_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_cluster_summary(answers):
    texts = [
        normalize_text(answer.get("raw_text", ""))
        for answer in answers
        if normalize_text(answer.get("raw_text", ""))
    ]
    return " ".join(texts)


def derive_reason(best_match, required_elements):
    matched_text = best_match["variation_text"]
    matched_elements = []
    missing_elements = []

    for element in required_elements:
        element_text = normalize_text(element)
        if not element_text:
            continue

        similarity = SequenceMatcher(
            None,
            element_text.lower(),
            matched_text.lower()
        ).ratio()

        if similarity >= 0.35:
            matched_elements.append(element_text)
        else:
            missing_elements.append(element_text)

    matched_part = ", ".join(matched_elements[:3]) if matched_elements else "core expected concepts"
    reason = (
        f"Best semantic match was variation {best_match['variation_id']} "
        f"with similarity {best_match['similarity_score']:.2f}, covering {matched_part}."
    )

    if missing_elements:
        reason += f" Missing or weaker areas: {', '.join(missing_elements[:2])}."

    return reason


def grade_from_similarity(similarity):
    if similarity >= 0.85:
        return 1.0
    if similarity >= 0.70:
        return 0.75
    if similarity >= 0.50:
        return 0.5
    return 0.0


def confidence_from_scores(sorted_scores):
    best = sorted_scores[0]["similarity_score"]
    second = sorted_scores[1]["similarity_score"] if len(sorted_scores) > 1 else 0.0
    margin = max(0.0, best - second)
    return round(min(1.0, best * 0.8 + margin * 0.2), 2)


def request_manual_review(cluster_payload, semantic_evaluation):
    print("\n--- MANUAL REVIEW REQUIRED ---")
    print(f"Cluster: {cluster_payload['cluster_id']}")
    print(f"Question: {cluster_payload.get('question_id', 'Unknown')}")
    print(f"Suggested marks: {semantic_evaluation['suggested_marks_display']}")
    print(f"Similarity threshold passed: {semantic_evaluation['passed_similarity_threshold']}")
    print(f"Confidence: {semantic_evaluation['confidence']}")
    print("Similarity scores:")

    for item in semantic_evaluation["variation_similarity_scores"]:
        print(
            f"  Variation {item['variation_id']}: "
            f"{item['similarity_score']:.2f}"
        )

    print(f"Reason: {semantic_evaluation['reason']}")

    try:
        prompt = (
            f"Enter manual marks (0-{semantic_evaluation['total_marks']})"
            if not semantic_evaluation["passed_similarity_threshold"]
            else f"Enter manual marks override (0-{semantic_evaluation['total_marks']}) or press Enter to accept"
        )
        manual_grade = input(f"{prompt}: ").strip()
        manual_reason = input(
            "Enter manual reason or press Enter to keep suggested reason: "
        ).strip()
    except EOFError:
        manual_grade = ""
        manual_reason = ""

    semantic_evaluation["manual_reviewed"] = False

    if manual_grade:
        try:
            parsed_marks = float(manual_grade)
            parsed_marks = max(0.0, min(float(semantic_evaluation["total_marks"]), parsed_marks))
            semantic_evaluation["suggested_marks"] = round(parsed_marks, 2)
            semantic_evaluation["suggested_marks_display"] = (
                f"{semantic_evaluation['suggested_marks']}/{semantic_evaluation['total_marks']}"
            )
            semantic_evaluation["manual_reviewed"] = True
        except ValueError:
            pass

    if manual_reason:
        semantic_evaluation["reason"] = manual_reason
        semantic_evaluation["manual_reviewed"] = True

    if not semantic_evaluation["passed_similarity_threshold"] and not manual_grade:
        semantic_evaluation["suggested_marks"] = None
        semantic_evaluation["suggested_marks_display"] = f"manual review required/{semantic_evaluation['total_marks']}"

    return semantic_evaluation


def evaluate_cluster_semantics(cluster_payload, rubric):
    semantic_variations = [
        normalize_text(text)
        for text in rubric.get("semantic_variations", [])
        if normalize_text(text)
    ]
    cluster_summary = build_cluster_summary(cluster_payload["answers"])

    if not semantic_variations or not cluster_summary:
        return {
            "cluster_summary": cluster_summary,
            "variation_similarity_scores": [],
            "best_variation": None,
            "suggested_marks": 0,
            "suggested_marks_display": "0/0",
            "total_marks": 0,
            "reason": "No semantic comparison could be computed.",
            "confidence": 0.0,
            "manual_reviewed": False
        }

    model = get_semantic_model()
    cluster_embedding = model.encode(cluster_summary, convert_to_tensor=True)
    variation_embeddings = model.encode(semantic_variations, convert_to_tensor=True)
    similarity_scores = util.cos_sim(cluster_embedding, variation_embeddings)[0].tolist()

    variation_score_rows = []
    for index, (variation_text, score) in enumerate(
        zip(semantic_variations, similarity_scores),
        start=1
    ):
        variation_score_rows.append({
            "variation_id": index,
            "variation_text": variation_text,
            "similarity_score": round(float(score), 4)
        })

    variation_score_rows.sort(
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    best_match = variation_score_rows[0]
    total_marks = rubric.get("total_marks", 0)
    passed_similarity_threshold = best_match["similarity_score"] > 0.50

    if passed_similarity_threshold:
        mark_ratio = grade_from_similarity(best_match["similarity_score"])
        suggested_marks = round(total_marks * mark_ratio, 2)
        suggested_marks_display = f"{suggested_marks}/{total_marks}"
    else:
        suggested_marks = None
        suggested_marks_display = f"manual review required/{total_marks}"
    reason = derive_reason(
        best_match,
        rubric.get("required_elements", [])
    )

    if not passed_similarity_threshold:
        reason = (
            f"Low semantic similarity detected ({best_match['similarity_score']:.2f}). "
            f"Assigned a conservative mark using required-element coverage. {reason}"
        )

    evaluation = {
        "cluster_summary": cluster_summary,
        "variation_similarity_scores": variation_score_rows,
        "best_variation": best_match,
        "suggested_marks": suggested_marks,
        "suggested_marks_display": suggested_marks_display,
        "total_marks": total_marks,
        "reason": reason,
        "confidence": confidence_from_scores(variation_score_rows),
        "passed_similarity_threshold": passed_similarity_threshold,
        "manual_reviewed": False
    }

    evaluation = request_manual_review(cluster_payload, evaluation)

    return evaluation


def grade_cluster(cluster_payload):
    start_time = time.time()
    results = []
    rubric = cluster_payload["rubric"]
    semantic_evaluation = evaluate_cluster_semantics(cluster_payload, rubric)

    for ans in cluster_payload["answers"]:
        student_text = ans["raw_text"].strip().replace("\n", " ")

        if not student_text or not re.search(r"[a-zA-Z]", student_text):
            results.append({
                "student_id": ans["student_id"],
                "score": 0,
                "confidence": 0,
                "feedback": ["Invalid or empty answer"],
                "suggested_marks": semantic_evaluation["suggested_marks_display"],
                "suggested_reason": semantic_evaluation["reason"],
                "suggested_confidence": semantic_evaluation["confidence"]
            })
            continue

        q_type = rubric.get("type", "theory")

        if q_type == "theory":
            required = rubric.get("required_elements", [])

            if not required:
                final_score = 0.3
            else:
                total_score = 0

                for concept in required:
                    similarity = SequenceMatcher(
                        None,
                        concept.lower(),
                        student_text.lower()
                    ).ratio()

                    if similarity > 0.7:
                        total_score += 1.0
                    elif similarity > 0.5:
                        total_score += 0.7
                    elif similarity > 0.3:
                        total_score += 0.4
                    else:
                        total_score += 0

                final_score = total_score / len(required)

        elif q_type == "math":
            student_math = extract_math_expressions(student_text)
            correct_eq = rubric.get("equation")

            if not student_math:
                final_score = 0
            else:
                correct = any(
                    validate_equation(expr, correct_eq)
                    for expr in student_math
                )
                final_score = 1.0 if correct else 0.5

        elif q_type == "language":
            similarity = SequenceMatcher(
                None,
                student_text.lower(),
                rubric.get("model_answer", "").lower()
            ).ratio()

            final_score = min(1.0, similarity * 1.2)

        else:
            final_score = 0

        if final_score >= 0.7:
            feedback = ["Excellent answer"]
        elif final_score >= 0.5:
            feedback = ["Good answer"]
        elif final_score >= 0.25:
            feedback = ["Partially correct answer"]
        else:
            feedback = ["Needs improvement"]

        results.append({
            "student_id": ans["student_id"],
            "score": round(final_score, 2),
            "confidence": round(final_score, 2),
            "feedback": feedback,
            "suggested_marks": semantic_evaluation["suggested_marks_display"],
            "suggested_reason": semantic_evaluation["reason"],
            "suggested_confidence": semantic_evaluation["confidence"]
        })

    avg_score = sum(r["score"] for r in results) / len(results)

    all_feedback = []
    for r in results:
        all_feedback.extend(r["feedback"])

    top_issues = [
        issue for issue, _ in Counter(all_feedback).most_common(3)
    ]

    return {
        "cluster_id": cluster_payload["cluster_id"],
        "question_id": cluster_payload.get("question_id"),
        "cluster_size": len(results),
        "avg_score": round(avg_score, 2),
        "semantic_evaluation": semantic_evaluation,
        "top_issues": top_issues,
        "processing_time": round(time.time() - start_time, 2),
        "results": results
    }
