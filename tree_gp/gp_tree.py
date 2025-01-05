# gp_tree.py
import random
import math
from node import Node
import operators

class GPTree:
    """
    Represents a genetic programming tree, 
    used to encode a candidate solution (symbolic expression).
    """

    def __init__(self, root=None):
        self.root = root

    def copy(self):
        """Return a deep copy of the current tree."""
        return GPTree(root=self.root.copy())

    def evaluate(self, x):
        """
        Evaluate the tree for a given input x (which can be a list/tuple of variables).
        x: list of input variables [x0, x1, ...].
        """
        return self._eval_node(self.root, x)

    def _eval_node(self, node, x):
        """Recursively evaluate a node."""
        if node.is_terminal():
            # Terminal is either x[i] or constant
            if isinstance(node.value, str) and node.value.startswith('x['):
                # e.g. 'x[0]' or 'x[1]'
                index = int(node.value[2:-1])  # get the index from 'x[i]'
                return x[index]
            else:
                # It's a constant
                return float(node.value)
        else:
            # Operator node
            if node.arity == 1:
                # Unary operator
                child_val = self._eval_node(node.children[0], x)
                return operators.UNARY_OPERATORS[node.value](child_val)
            elif node.arity == 2:
                # Binary operator
                left_val = self._eval_node(node.children[0], x)
                right_val = self._eval_node(node.children[1], x)
                return operators.BINARY_OPERATORS[node.value](left_val, right_val)

    @staticmethod
    def generate_random(depth, variables_count, max_depth, full=False):
        """
        Generate a random tree using either 'grow' or 'full' method.
        depth: current depth in the recursion
        variables_count: number of available variables
        max_depth: maximum allowed depth
        full: if True, use full method, otherwise grow method
        """
        if depth >= max_depth:
            return GPTree._random_terminal(variables_count)

        # Decide whether to create an operator node or a terminal node
        # If using 'full' method: always create operator until max depth
        # If using 'grow' method: randomly choose
        if (full and depth < max_depth) or (not full and random.random() < 0.7):
            op, arity = operators.random_operator()
            children = []
            for _ in range(arity):
                child_tree = GPTree.generate_random(depth+1, variables_count, max_depth, full)
                children.append(child_tree.root)
            root_node = Node(op, arity, children)
            return GPTree(root=root_node)
        else:
            return GPTree._random_terminal(variables_count)

    @staticmethod
    def _random_terminal(variables_count):
        """
        Returns a tree with a single terminal node.
        Terminal can be either x[i] or a random constant.
        """
        if random.random() < 0.5:
            # Use a variable
            var_index = random.randint(0, variables_count - 1)
            node = Node(f'x[{var_index}]', arity=0, children=[])
        else:
            # Use a random constant
            node = Node(random.uniform(-2.0, 2.0), arity=0, children=[])
        return GPTree(root=node)
