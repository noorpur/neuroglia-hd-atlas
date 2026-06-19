# Dataset Card

## Purpose

This repository analyses public Huntingtonian neuroglial transcriptomic datasets using cell-type-aware machine learning, pseudobulk aggregation, pathway scoring, leakage-aware validation, and latent-state modelling.

## Dataset registry

| Dataset ID | Accession | Organism | Modality | Role |
|---|---:|---|---|---|
| `human_glia_sn` | `GSE281069` | Human | snRNA-seq | Primary human glial atlas |
| `mouse_r62_sn` | `GSE180294` | Mouse | snRNA-seq | Cross-species and region-stress testing |
| `human_bulk_cag` | `GSE159940` | Human | bulk rdRNA-seq | CAG and clinical-variable context |
| `human_bulk_ba9` | `GSE64810` | Human | bulk mRNA-seq | Independent human replication |

## Access

The pipeline downloads supplementary files from the NCBI GEO FTP hierarchy. Raw files are not committed because they are large and should remain governed by original data-use terms.

## Intended use

- exploratory disease-state modelling;
- mechanism-aware feature analysis;
- donor-level validation design;
- candidate gene/signature prioritisation for follow-up experiments.

## Not intended use

- clinical diagnosis;
- treatment recommendation;
- automated patient triage;
- claims of causality from observational expression alone.

## Known limitations

- post-mortem tissue reflects end-stage or advanced molecular context;
- single-nucleus data may have donor, region, and dissociation/preparation effects;
- cell annotations may differ across studies;
- mouse-to-human transfer is biologically informative but not direct equivalence;
- expression of modifier genes is not the same as inherited modifier genotype or somatic repeat measurement.
