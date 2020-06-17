import libsbn

def get_instance(newick_file, name='treeflow'):
    inst = libsbn.rooted_instance(name)
    inst.read_newick_file(newick_file)
    return inst