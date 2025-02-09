# operators.py
import numpy as np
import random

def safe_log(x):
    return np.log(np.clip(x, 1e-5, None))  # Clip values lower than 1e-5


def safe_exp(x):
    # Clip x to avoid overflow.
    return np.exp(np.clip(x, -100, 100))

def safe_sqrt(x):
    # Ensure nonnegative values.
    return np.sqrt(np.clip(x, 0, None))

def safe_div(x, y):
    # Use np.divide with a where mask to avoid division by very small numbers.
    # Where y is very small, return 1.0.
    return np.where(np.abs(y) < 1e-10, 1.0, x / y)

UNARY_OPERATORS = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'log': safe_log,
    'exp': safe_exp,
    'sqrt': safe_sqrt,
    'negate': lambda x: -x,
    # 'square': lambda x: np.clip(x**2, -1e308, 1e308),
     'cube': lambda x: np.clip(x**3, -1e308, 1e308),
}

BINARY_OPERATORS = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': safe_div,
}

def random_operator():
    # 80% chance to pick a binary operator, 20% chance for a unary.
    if random.random() < 0.8:
        op = random.choice(list(BINARY_OPERATORS.keys()))
        arity = 2
    else:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        arity = 1
    return op, arity