# visualize.py

from graphviz import Digraph

def visualize_tree(gp_tree, filename="best_tree.gv"):
    """
    Visualize the tree using Graphviz.
    """
    dot = Digraph(comment='GP Tree')
    _add_nodes(dot, gp_tree.root)
    dot.render(filename, view=False, format='png')  # Renders to .gv + .png

def _add_nodes(dot, node, parent_id=None, counter=[0]):
    """
    Recursive helper function to traverse the tree and add nodes/edges to the graph.
    'counter' is a list to simulate pass-by-reference for indexing nodes uniquely.
    """
    node_id = str(counter[0])
    dot.node(node_id, str(node.value))
    counter[0] += 1

    if parent_id is not None:
        dot.edge(parent_id, node_id)

    for child in node.children:
        _add_nodes(dot, child, node_id, counter)
