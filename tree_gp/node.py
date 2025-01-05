from operators import UNARY_OPERATORS, BINARY_OPERATORS

class Node: 
    def __init__(self, value, arity=0, children = None):
        self.value = value
        self.arity = arity # 0 for leaf nodes, 1 for unary and 2 for binary
        self.children = children if children is not None else []
        
        # here we validate the node structure
        assert len(self.children) == self.arity, (
            f"Node '{self.value}' expected {self.arity} children, "
            f"but got {len(self.children)}"
        )
        
    def is_leaf(self):
        return self.arity == 0  # has no children means leaf node

    
    
    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"{self.value} -> [{', '.join(str(child) for child in self.children)}]"

    def __call__(self, **variables):
        # if its a constant or variable node
        if self.is_leaf():
            if isinstance(self.value, str) and self.value.startswith("x"):
                # handle variables (e.g. "x_0")
                return variables[self.value]
            else: 
                # handle constant
                return self.value
        else:# if its an operator 
        # Evaluate children
            child_values = [child(**variables) for child in self.children]
            from operators import UNARY_OPERATORS, BINARY_OPERATORS
            if self.arity == 1:  # Unary operator
                return UNARY_OPERATORS[self.value](child_values[0])
            elif self.arity == 2:  # Binary operator
                return BINARY_OPERATORS[self.value](*child_values)
            else:
                raise ValueError(f"Unsupported arity for operator '{self.value}'")

    def get_subtree(self):
        # return all nodes in the subtree
        pass
    
    def mutate(self):
        # here we can modify the current node or the children nodes for mutation
        pass
    
    def validate(self):
        # validate the node structure 
        pass
    
    def __str__(self):
        """
        String representation of the node and its subtree.
        """
        if self.is_leaf():
            return str(self.value)
        else:
            children_str = ", ".join(str(child) for child in self.children)
            return f"{self.value}({children_str})"
        
    
    
