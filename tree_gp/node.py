import numpy as np
from operators import UNARY_OPERATORS, BINARY_OPERATORS

class Node:
    def __init__(self, value, arity=0, children=None):
        self.value = value
        self.arity = arity
        self.children = children if children is not None else []
        assert len(self.children) == self.arity, (
            f"Node '{self.value}' expected {self.arity} children, "
            f"but got {len(self.children)}"
        )

    def is_leaf(self):
        return self.arity == 0

    def copy(self):
        return Node(
            self.value,
            self.arity,
            [child.copy() for child in self.children]
        )

    def evaluate(self, variables):
        """Non-vectorized evaluation of a single sample (for reference)."""
        if self.is_leaf():
            # variable or constant
            if isinstance(self.value, str) and self.value.startswith("x_"):
                index = int(self.value.split("_")[1])
                return variables[index]
            else:
                return self.value
        else:
            # Evaluate child nodes
            child_vals = [child.evaluate(variables) for child in self.children]
            if self.arity == 1:
                return UNARY_OPERATORS[self.value](child_vals[0])
            else:
                return BINARY_OPERATORS[self.value](child_vals[0], child_vals[1])

    def evaluate_vectorized(self, X):
        """
        Vectorized evaluation for all samples in X (shape=(n_features, n_samples)).
        Returns a 1D NumPy array of length n_samples.
        """
        if self.is_leaf():
            if isinstance(self.value, str) and self.value.startswith("x_"):
                # variable
                var_idx = int(self.value.split("_")[1])
                return X[var_idx, :]
            else:
                # constant
                return np.full(X.shape[1], self.value)
        else:
            child_vals = [child.evaluate_vectorized(X) for child in self.children]
            if self.arity == 1:
                return UNARY_OPERATORS[self.value](child_vals[0])
            else:
                return BINARY_OPERATORS[self.value](child_vals[0], child_vals[1])

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        else:
            return f"{self.value}(" + ", ".join(str(c) for c in self.children) + ")"
