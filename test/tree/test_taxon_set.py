import typing as tp
from treeflow.tree.taxon_set import DictTaxonSet, TaxonSet, TupleTaxonSet

taxa = {"a", "b", "c"}


def taxon_set_checks(TaxonSetClass: tp.Callable[[tp.Iterable[str]], TaxonSet]):
    taxon_set = TaxonSetClass(taxa)
    assert set(taxon_set) == taxa
    assert "a" in taxon_set
    assert not ("d" in taxon_set)
    assert len(taxon_set) == 3


def test_dict_taxon_set():
    taxon_set_checks(DictTaxonSet)


def test_tuple_taxon_set():
    taxon_set_checks(TupleTaxonSet)


def test_taxon_set_conversion():
    conversion_taxon_set = lambda taxa: TupleTaxonSet(DictTaxonSet(taxa))
    taxon_set_checks(conversion_taxon_set)
