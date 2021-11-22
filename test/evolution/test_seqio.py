from treeflow.evolution.seqio import Alignment


def test_seqio_parse_fasta(hello_fasta_file):
    alignment = Alignment(hello_fasta_file)
    expected_keys = {"mars", "saturn", "jupiter"}
    expected_len = 31
    assert set(alignment.sequence_mapping.keys()) == expected_keys
    for key in expected_keys:
        assert (len(alignment.sequence_mapping[key])) == expected_len
