import numpy as np
import random

UNARY_OPERATORS = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'log': lambda x: np.where(x <= 0, 0, np.log(x)),  # Safe log: clamp to 0 if x <= 0
    'exp': lambda x: np.where(x > 709, np.inf, np.exp(x)),  # Avoid overflow for large x
    'square': lambda x: np.clip(x ** 2, -1e308, 1e308),  # Clamp square to prevent overflow
    'cube': lambda x: np.clip(x ** 3, -1e308, 1e308),  # Clamp cube to prevent overflow
    'sqrt': lambda x: np.where(x < 0, 0, np.sqrt(x)),  # Safe sqrt: clamp negatives to 0
    'asin': lambda x: np.arcsin(np.clip(x, -1, 1)),  # Clamp to [-1, 1] for domain
    'acos': lambda x: np.arccos(np.clip(x, -1, 1)),  # Clamp to [-1, 1] for domain
    'atan': np.arctan,
    'negate': lambda x: -x
}

BINARY_OPERATORS = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: np.where(np.abs(y) < 1e-10, 1.0, x / y),  # Safe division: avoid divide by zero
    'pow': lambda x, y: np.where((x < 0) & (y < 1), 0, np.clip(x ** y, -1e308, 1e308)),  # Safe pow
    'mod': lambda x, y: np.where(np.abs(y) < 1e-10, 0, x % y),  # Safe mod: avoid divide by zero
}


def random_operator():
    """
    Choose an operator (unary or binary) randomly.
    Return (operator_symbol, arity).
    """
    # 80% chance for a binary operator, 20% chance for a unary
    if random.random() < 0.6:
        op = random.choice(list(BINARY_OPERATORS.keys()))
        arity = 2
    else:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        arity = 1
    return op, arity
