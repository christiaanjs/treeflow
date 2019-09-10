from beagle_likelihood import BeagleLikelihood

FASTA_FILE = 'data/sim-seq.fasta'
NEWICK_FILE = 'data/analysis-tree.nwk'

likelihood_classes = [BeagleLikelihood]

for cls in likelihood_classes:
    likelihood = cls(newick_file=NEWICK_FILE, fasta_file=FASTA_FILE)
