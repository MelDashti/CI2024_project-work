import numpy as np
import random

UNARY_OPERATORS = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    # Safe log: clamp to 0 if x <= 0
    'log': lambda x: np.where(x <= 0, 0, np.log(x))
}

BINARY_OPERATORS = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    # Safe division: if y=0, return 1.0 (element-wise)
    '/': lambda x, y: np.where(y == 0, 1.0, x / y),
}

def random_operator():
    """
    Choose an operator (unary or binary) randomly.
    Return (operator_symbol, arity).
    """
    # 80% chance for a binary operator, 20% chance for a unary
    if random.random() < 0.8:
        op = random.choice(list(BINARY_OPERATORS.keys()))
        arity = 2
    else:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        arity = 1
    return op, arity
