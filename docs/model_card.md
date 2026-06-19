# Model Card

## Model name

NeuroGlia-HD Atlas modelling suite

## Model type

A collection of leakage-aware classifiers and latent representation models for transcriptomic disease-state analysis.

## Input

Pseudobulk expression and curated signature features derived from public human and mouse transcriptomic datasets.

## Output

- probability-like disease-state scores;
- latent embeddings;
- feature importance tables;
- ablation summaries;
- calibration metrics;
- audit tables.

## Primary metrics

- ROC-AUC
- average precision
- balanced accuracy
- Brier score
- calibration curves
- ablation deltas

## Ethical and safety boundaries

The models are not clinical tools. They are research instruments for hypothesis generation and computational prioritisation. Any biological interpretation requires independent dataset replication and experimental validation.

## Failure modes

- hidden donor or preparation batch effects;
- inflated accuracy from metadata leakage;
- overinterpretation of expression signatures as causality;
- cross-species mismatch;
- cell annotation inconsistencies.

## Risk controls

- grouped validation by donor;
- held-out region testing;
- signature coverage reports;
- null-label controls;
- explicit model limitations in generated reports.
