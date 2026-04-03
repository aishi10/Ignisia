from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
from sympy import simplify
import re

# Enable implicit multiplication (2m → 2*m, mv → m*v)
transformations = standard_transformations + (implicit_multiplication_application,)


def normalize_expression(expr):
    expr = expr.lower()

    # Remove spaces
    expr = expr.replace(" ", "")

    # Replace power symbol
    expr = expr.replace("^", "**")

    # Handle fractions like ½
    expr = expr.replace("½", "1/2")

    # Replace common words → symbols
    expr = expr.replace("mass", "m")
    expr = expr.replace("velocity", "v")
    expr = expr.replace("energy", "e")

    # 🚨 IMPORTANT: remove units WITHOUT removing variables
    expr = re.sub(r'(m/s|kg|n|j|cm)', '', expr)

    # Take RHS if equation
    if "=" in expr:
        expr = expr.split("=")[-1]

    return expr.strip()


def validate_equation(student_expr, correct_expr):
    try:
        if not student_expr:
            return False

        student_expr = normalize_expression(student_expr)
        correct_expr = normalize_expression(correct_expr)

        # 🔥 NO local_dict, NO symbols → let sympy handle it
        student = simplify(parse_expr(student_expr, transformations=transformations))
        correct = simplify(parse_expr(correct_expr, transformations=transformations))

        return simplify(student - correct) == 0

    except Exception as e:
        print("Math error:", e)
        return False
