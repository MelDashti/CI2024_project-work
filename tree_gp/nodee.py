class nodee: 
    def __init__(self, value, children = None):
        self.value = value
        self.children = children or []
        
    def is_leaf(self):
        return len(self.children) == 0  # has no children means leaf node

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"{self.value} -> [{', '.join(str(child) for child in self.children)}]"

    def __call__(self, *args, **kwds):
        pass
        # evaluate the node recursively

    def get_subtree(self):
        # return all nodes in the subtree
        pass
    
    def mutate(self):
        # here we can modify the current node or the children nodes for mutation
        pass
    
    def validate(self):
        # validate the node structure 
        pass

        
    
    
if __name__ == '__main__':
 # Create a three-level tree
    x_node = nodee("x_0")
    const_node1 = nodee(5)
    const_node2 = nodee(3)
    add_node1 = nodee("+", [x_node, const_node1])
    add_node2 = nodee("+", [add_node1, const_node2])
    print(add_node2)