from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    accession: str
    title: str
    organism: str
    modality: str
    role: str
    primary_url: str
    notes: str


DATASETS: dict[str, DatasetRecord] = {
    "human_glia_sn": DatasetRecord(
        dataset_id="human_glia_sn",
        accession="GSE281069",
        title="Mapping the glial transcriptome in Huntington's disease using snRNAseq",
        organism="Homo sapiens",
        modality="single-nucleus RNA-seq",
        role="primary human glial atlas",
        primary_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE281069",
        notes="Human post-mortem brain regions with HD and matched control donors; useful for astrocyte, microglia, OPC and oligodendrocyte state modelling.",
    ),
    "mouse_r62_sn": DatasetRecord(
        dataset_id="mouse_r62_sn",
        accession="GSE180294",
        title="Single nucleus RNA sequencing of nontransgenic and Huntington's disease R6/2 mouse striatum and cortex",
        organism="Mus musculus",
        modality="single-nucleus RNA-seq",
        role="cross-species transfer and region-stress testing",
        primary_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE180294",
        notes="Mouse R6/2 cortex and striatum; useful for testing whether glial-state signatures transfer beyond the human cohort.",
    ),
    "human_bulk_cag": DatasetRecord(
        dataset_id="human_bulk_cag",
        accession="GSE159940",
        title="Ribo-depleted RNA-Seq libraries from Huntington's disease brain",
        organism="Homo sapiens",
        modality="bulk rdRNA-seq",
        role="CAG and clinical-variable context",
        primary_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159940",
        notes="Post-mortem HD prefrontal cortex with disease variables including CAG repeat size and age of onset in study metadata.",
    ),
    "human_bulk_ba9": DatasetRecord(
        dataset_id="human_bulk_ba9",
        accession="GSE64810",
        title="mRNA-seq expression profiling of human post-mortem BA9 brain tissue for Huntington's disease and neurologically normal individuals",
        organism="Homo sapiens",
        modality="bulk mRNA-seq",
        role="independent human replication",
        primary_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64810",
        notes="Human BA9 expression data from HD and neurologically normal controls; useful for replication of inflammatory and developmental signatures.",
    ),
}


def list_records() -> list[DatasetRecord]:
    return list(DATASETS.values())


def get_record(dataset_id: str) -> DatasetRecord:
    try:
        return DATASETS[dataset_id]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset_id={dataset_id!r}. Valid values: {valid}") from exc
