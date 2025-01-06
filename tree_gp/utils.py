import numpy as np
import sympy
from sympy import Symbol, sympify

def load_dataset(filepath="data/problem_0.npz", test_split=0.2):
    data = np.load(filepath)
    x = data['x']  # shape (n_features, n_samples)
    y = data['y']  # shape (n_samples,)

    n_samples = y.shape[0]
    n_test = int(n_samples * test_split)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    x_train, x_test = x[:, indices[:-n_test]], x[:, indices[-n_test:]]
    y_train, y_test = y[indices[:-n_test]], y[indices[-n_test:]]
    return x_train, y_train, x_test, y_test

def mean_squared_error(gp_tree, X, y):
    preds = gp_tree.evaluate_vectorized(X)
    return np.mean((preds - y) ** 2)

def calculate_mse(predictions, true_values):
    return 100 * np.square(predictions - true_values).sum() / len(true_values)

# -----------------------------------------------------
# Symbolic simplification for the final expression
# -----------------------------------------------------

def tree_to_sympy(node):
    """
    Recursively convert a GP Node into a Sympy expression.
    Handles unary and binary operators robustly, ensuring all supported
    operators are mapped to their Sympy equivalents.
    """
    if node.is_leaf():
        if isinstance(node.value, str) and node.value.startswith("x_"):
            # Variable
            idx = int(node.value.split("_")[1])
            return Symbol(f'x{idx}', real=True)  # e.g., x0, x1, ...
        else:
            # Constant
            return sympy.Float(node.value)
    else:
        # Operator node
        unary_ops = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'log': sympy.log,
            'exp': sympy.exp,
            'sqrt': sympy.sqrt,
            'negate': lambda x: -x,
            'square': lambda x: x**2,
            'cube': lambda x: x**3,
            'asin': sympy.asin,
            'acos': sympy.acos,
            'atan': sympy.atan,
        }
        binary_ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            'pow': lambda x, y: x**y,
            'mod': lambda x, y: x % y,
        }

        if node.arity == 1:
            child_expr = tree_to_sympy(node.children[0])
            if node.value in unary_ops:
                return unary_ops[node.value](child_expr)
            else:
                raise ValueError(f"Unknown unary op: {node.value}")
        elif node.arity == 2:
            left_expr = tree_to_sympy(node.children[0])
            right_expr = tree_to_sympy(node.children[1])
            if node.value in binary_ops:
                return binary_ops[node.value](left_expr, right_expr)
            else:
                raise ValueError(f"Unknown binary op: {node.value}")
        else:
            raise ValueError(f"Unsupported operator arity: {node.arity}")

def simplify_tree(gp_tree):
    """
    Convert the final GPTree to a Sympy expression, then simplify.
    Return a simplified Sympy expression as a string.
    """
    expr = tree_to_sympy(gp_tree.root)
    try:
        simplified_expr = sympy.simplify(expr)
    except Exception as e:
        print(f"Error during simplification: {e}")
        simplified_expr = expr  # Fallback to the unsimplified expression
    return str(simplified_expr)