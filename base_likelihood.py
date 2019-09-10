import sbn

class BaseLikelihood:
    def __init__(self, newick_file, fasta_file):
        self.newick_file = newick_file
        self.fasta_file = fasta_file
        self.inst = sbn.instance('sbninstance')
        self.inst.read_fasta_file(fasta_file)
        self.inst.read_newick_file(newick_file)
        self.inst.print_status()
