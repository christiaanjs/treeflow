def extract_operations(cf):
    """Recursively extract all tf.Operation objects from a ConcreteFunction's
    graph, descending into PartitionedCall / StatefulPartitionedCall bodies.

    Returns a flat list of tf.Operation in pre-order (each call op immediately
    followed by the operations of its body).
    """
    CALL_OPS = ("PartitionedCall", "StatefulPartitionedCall")

    def resolve(graph, fname):
        if isinstance(fname, bytes):
            fname = fname.decode()
        seen = set()
        g = graph
        while g is not None and id(g) not in seen:
            seen.add(id(g))
            funcs = getattr(g, "_functions", {})
            if fname in funcs:
                fg = getattr(funcs[fname], "graph", None)
                if fg is not None:
                    return fg
            g = getattr(g, "outer_graph", None)
        return None

    def walk(graph, visited):
        ops = []
        for op in graph.get_operations():
            ops.append(op)
            if op.type in CALL_OPS:
                fname = op.get_attr("f").name
                key = fname.decode() if isinstance(fname, bytes) else fname
                if key not in visited:
                    sub = resolve(graph, fname)
                    if sub is not None:
                        ops.extend(walk(sub, visited | {key}))
        return ops

    return walk(cf.graph, frozenset())
