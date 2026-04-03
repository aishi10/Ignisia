import re

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    # Remove common useless words
    stopwords = {
        "the", "is", "are", "was", "were", "this", "that",
        "with", "from", "have", "has", "had", "will",
        "shall", "can", "could", "would", "should"
    }

    keywords = [w for w in words if w not in stopwords]

    # Remove duplicates
    return list(set(keywords))


def extract_equation(text):
    match = re.findall(r'[a-zA-Z]+\s*=\s*[a-zA-Z0-9\*\+\-/\^\.\(\)]+', text)
    return match[0] if match else None


def generate_rubric(reference_answer):
    keywords = extract_keywords(reference_answer)
    equation = extract_equation(reference_answer)

    return {
        "keywords": keywords,
        "optional_keywords": [],
        "keyword_weight": 0.5,
        "equation": equation,
        "math_weight": 0.3
    }
