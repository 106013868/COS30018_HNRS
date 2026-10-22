"""
arithmetic.py
calculates the result of a recognised expression like "12+3*4"
"""

# doesnt use eval() since eval runs any python code it is given
# instead the expression is read one character at a time with the usual order of operations:
#   brackets first, then * and /, then + and -


def calculate(expression):
    """
    In: expression string made of digits and + - * / ( )
    Out: result as a number, raises ValueError if the expression is invalid
    """
    tokens = tokenise(expression)
    result, pos = read_sum(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"Unexpected '{tokens[pos]}' in {expression}")
    return result


def tokenise(expression):
    """
    splits expression into numbers and symbols, e.g. "12+3" -> [12, "+", 3]
    """
    tokens = []
    number = ""
    for char in expression:
        if char.isdigit():
            number += char
            continue
        if number:
            tokens.append(int(number))
            number = ""
        if char not in "+-*/()":
            raise ValueError(f"Unknown symbol '{char}'")
        tokens.append(char)
    if number:
        tokens.append(int(number))
    return tokens


def read_sum(tokens, pos):
    """
    reads terms joined by + or -, returns value and next position
    """
    value, pos = read_product(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("+", "-"):
        op = tokens[pos]
        right, pos = read_product(tokens, pos + 1)
        value = value + right if op == "+" else value - right
    return value, pos


def read_product(tokens, pos):
    """
    reads values joined by * or /, returns value and next position
    """
    value, pos = read_value(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("*", "/"):
        op = tokens[pos]
        right, pos = read_value(tokens, pos + 1)
        if op == "*":
            value = value * right
        elif right == 0:
            raise ValueError("Division by zero")
        else:
            value = value / right
    return value, pos


def read_value(tokens, pos):
    """
    reads a number, a bracketed expression or a negative value
    """
    if pos >= len(tokens):
        raise ValueError("Expression ends too early")
    token = tokens[pos]

    if isinstance(token, int):
        return token, pos + 1

    # minus at the start of a value means negative, e.g. "-3" or "2*-3"
    if token == "-":
        value, pos = read_value(tokens, pos + 1)
        return -value, pos

    if token == "(":
        value, pos = read_sum(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("Missing closing bracket")
        return value, pos + 1

    raise ValueError(f"Unexpected '{token}'")
