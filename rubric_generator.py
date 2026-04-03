import re

def _dedupe_preserve_order(items):
    seen = set()
    ordered = []

    for item in items:
        text = str(item or "").strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        ordered.append(text)

    return ordered


def generate_semantic_variations(model_answer, required_elements):
    cleaned_elements = [element.strip() for element in required_elements if element.strip()]
    variations = [model_answer]

    if cleaned_elements:
        variations.append(". ".join(cleaned_elements) + ".")
        variations.append(
            "Key points: " + "; ".join(cleaned_elements) + "."
        )

        first_half = cleaned_elements[:max(1, len(cleaned_elements) // 2)]
        second_half = cleaned_elements[max(1, len(cleaned_elements) // 2):]

        variations.append(
            "Expected answer should mention " + ", ".join(first_half) + "."
        )

        if second_half:
            variations.append(
                "A correct response also includes " + ", ".join(second_half) + "."
            )

    return _dedupe_preserve_order(variations)


def extract_total_marks(question_text, required_elements):
    question_text = str(question_text or "")
    marks_patterns = [
        r"(\d+)\s*marks?",
        r"\((\d+)\)",
        r"\[(\d+)\]"
    ]

    for pattern in marks_patterns:
        match = re.search(pattern, question_text, re.IGNORECASE)
        if match:
            return max(1, int(match.group(1)))

    if required_elements:
        return max(1, len(required_elements))

    return 3


def detect_question_type(text, model_answer, required_elements):
    combined = str(text or "").lower()
    model_lower = str(model_answer or "").lower()
    required_lower = " ".join(required_elements).lower()

    code_markers = [
        "input(",
        "print(",
        "if ",
        "else",
        "elif",
        "for ",
        "while ",
        "def ",
        "return",
        "int(",
        "float(",
        "scanf",
        "printf",
        "python",
        "program",
        "code",
        "syntax"
    ]

    if any(marker in combined for marker in code_markers):
        return "language"

    math_patterns = [
        r"^\s*[a-z0-9_]+\s*=\s*[0-9a-z_+\-*/^(). ]+\s*$",
        r"\bsolve\b",
        r"\bsimplify\b",
        r"\bcalculate\b",
        r"\bevaluate\b",
        r"\bequation\b"
    ]

    if any(re.search(pattern, combined, re.IGNORECASE) for pattern in math_patterns):
        if not any(marker in model_lower or marker in required_lower for marker in code_markers):
            return "math"

    if len(combined.split()) > 25:
        return "language"

    return "theory"


def generate_rubric(teacher_data):

    # -----------------------------
    # SAFE EXTRACTION
    # -----------------------------
    model_answer = ""
    required = ""
    question_text = ""

    if isinstance(teacher_data, dict):
        question_text = teacher_data.get("question", "") or ""
        model_answer = teacher_data.get("model_answer", "") or ""
        required = teacher_data.get("required", "") or ""
    else:
        model_answer = str(teacher_data)

    text = (model_answer + " " + required).lower()

    # -----------------------------
    # REQUIRED ELEMENTS
    # -----------------------------
    parts = re.split(r'\d+\.\s*', required)
    required_elements = [p.strip() for p in parts if p.strip()]

    # fallback if empty
    if not required_elements:
        words = re.findall(r'[a-zA-Z]+', text)
        required_elements = list(set(words))[:3]

    q_type = detect_question_type(text, model_answer, required_elements)

    semantic_variations = generate_semantic_variations(
        model_answer,
        required_elements
    )
    total_marks = extract_total_marks(question_text, required_elements)

    return {
        "type": q_type,
        "required_elements": required_elements,
        "model_answer": model_answer,
        "equation": model_answer if q_type == "math" else None,
        "semantic_variations": semantic_variations,
        "total_marks": total_marks
    }
