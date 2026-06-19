from neuroglia_hd.data.geo import geo_series_prefix, geo_supplementary_url
from neuroglia_hd.data.registry import get_record, list_records


def test_registry_contains_expected_accessions():
    accessions = {r.accession for r in list_records()}
    assert {"GSE281069", "GSE180294", "GSE159940", "GSE64810"}.issubset(accessions)


def test_geo_prefix_builder():
    assert geo_series_prefix("GSE281069") == "GSE281nnn"
    assert geo_supplementary_url("GSE281069").endswith("/GSE281069/suppl/")
    assert get_record("human_glia_sn").organism == "Homo sapiens"
