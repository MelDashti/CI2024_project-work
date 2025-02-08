import random
from node import Node
import operators

class GPTree:
    def __init__(self, root=None):
        self.root = root

    def copy(self):
        return GPTree(self.root.copy())

    def evaluate(self, variables):
        """Evaluate for one sample (non-vectorized)."""
        return self.root.evaluate(variables)

    def evaluate_vectorized(self, X):
        """Vectorized evaluation for all samples in X."""
        return self.root.evaluate_vectorized(X)

    @staticmethod
    def generate_random(depth, variables_count, max_depth, full=False):
        # If we reached max depth, return a terminal
        if depth >= max_depth:
            return GPTree._random_terminal(variables_count)

        # 70% chance to choose operator if not full method
        if (full and depth < max_depth) or (not full and random.random() < 0.7):
            op, arity = operators.random_operator()
            children = []
            for _ in range(arity):
                child_tree = GPTree.generate_random(
                    depth + 1, variables_count, max_depth, full
                )
                children.append(child_tree.root)
            return GPTree(root=Node(op, arity, children))
        else:
            return GPTree._random_terminal(variables_count)

    @staticmethod
    def _random_terminal(variables_count):
        # 50% variable, 50% constant
        if random.random() < 0.5:
            var_index = random.randint(0, variables_count - 1)
            node = Node(f'x_{var_index}', 0, [])
        else:
            node = Node(random.uniform(-2.0, 2.0), 0, [])
        return GPTree(root=node)
