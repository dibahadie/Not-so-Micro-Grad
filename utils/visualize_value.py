from graphviz import Digraph

def trace(root):
    """Trace a computational graph starting from the root variable.

    Args:
        root: The root variable from which to start tracing.

    Returns:
        A set of all variables and functions in the computational graph.
    """
    nodes, edges = set(), set()
    def add_nodes(var):
        if var not in nodes:
            nodes.add(var)
            for child in var._prev:
                edges.add((child, var))
                add_nodes(child)
    add_nodes(root)
    return nodes, edges


def visualize(root):
    """Visualize the computational graph starting from the root variable.

    Args:
        root: The root variable from which to start visualization.

    Returns:
        A Graphviz Digraph object representing the computational graph.
    """
    nodes, edges = trace(root)
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    for n in nodes:
        uid = str(id(n))
        label = f"{n.expression} | data {n.data:.4f} | grad {n.grad:.4f}"
        dot.node(name=uid, label=label, shape='box')

        if n._op:
            dot.node(name=uid + n._op, label=n._op, shape='ellipse')
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot