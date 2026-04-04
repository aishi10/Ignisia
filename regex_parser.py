import re

def extract_math_expressions(text):
    # Match full equations like p=m*v or E=mc^2
    pattern = r'[a-zA-Z]+\s*=\s*[a-zA-Z0-9\*\+\-/\^\.\(\)]+'
    
    matches = re.findall(pattern, text)

    # Clean spaces
    matches = [m.replace(" ", "") for m in matches]

    return matches
