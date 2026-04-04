from rapidfuzz import fuzz

def keyword_score(student_text, keywords):
    score = 0
    for word in keywords:
        match = fuzz.partial_ratio(word.lower(), student_text.lower())
        if match > 80:
            score += 1
    return score / len(keywords) if keywords else 0
