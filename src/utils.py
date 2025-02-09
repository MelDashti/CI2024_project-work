import numpy as np
import sympy
from sympy import Symbol, sympify

def load_dataset(filepath, test_split=0.2):
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


# In utils.py

import sympy
from sympy import Symbol, sympify

def tree_to_sympy(node):
    """
    Recursively convert a GP Node into a Sympy expression.
    Handles unary and binary operators robustly.
    """
    if node.is_leaf():
        if isinstance(node.value, str) and node.value.startswith("x_"):
            idx = int(node.value.split("_")[1])
            return Symbol(f'x{idx}', real=True)
        else:
            # Use sympify to try and convert numbers or strings to sympy types.
            return sympify(node.value)
    else:
        # Updated operator dictionaries with 'cube' and 'square'
        unary_ops = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'log': sympy.log,
            'exp': sympy.exp,
            'sqrt': sympy.sqrt,
            'negate': lambda x: -x,
            'cube': lambda x: x**3,
            'square': lambda x: x**2,
        }
        binary_ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
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
        # First clean up any problematic terms
        if expr.has(sympy.zoo) or expr.has(sympy.oo) or expr.has(sympy.nan):
            expr = expr.replace(sympy.zoo, 1e100)
            expr = expr.replace(sympy.oo, 1e100)
            expr = expr.replace(-sympy.oo, -1e100)
            expr = expr.replace(sympy.nan, 0)

        # Try different simplification strategies
        results = []
        
        # Basic expansion
        try:
            results.append(sympy.expand(expr))
        except:
            pass
            
        # Try the default simplify
        try:
            results.append(sympy.simplify(expr))
        except:
            pass
            
        # Try collecting terms
        try:
            results.append(sympy.collect(expr, expr.free_symbols))
        except:
            pass
            
        # Try factoring
        try:
            results.append(sympy.factor(expr))
        except:
            pass
        
        # Choose the simplest valid result
        valid_results = [r for r in results if r is not None]
        if valid_results:
            simplified_expr = min(valid_results, key=lambda x: x.count_ops())
            # If the simplified result is too complex, return original
            if simplified_expr.count_ops() > 2 * expr.count_ops():
                return str(expr)
            return str(simplified_expr)
        
        # If no simplification worked, return original
        return str(expr)
        
    except Exception as e:
        print(f"Error during simplification: {e}")
        return str(expr)  # fallback to unsimplified expression