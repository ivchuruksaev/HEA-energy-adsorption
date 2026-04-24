# HEA energy adsorption screening

This repository contains code and processed-data interfaces for a high-entropy alloy adsorption-energy screening workflow.

The workflow combines three levels of modelling:

1. DFT-derived adsorption energies for selected alloy configurations.
2. A frozen teacher model used to assign surrogate descriptor values to the full 19-metal, five-component composition space.
3. Interpretable linear-quadratic and graph-neural-network models used to analyse and reproduce the teacher-derived ranking.

## Repository structure

```text
docs/
  reproducibility.md
  data_dictionary.md

scripts/
  reproducibility_utils.py
  export_composition_tables.py
  run_lq_learning_curve.py

data/
  dft_calculated_alloys.csv
  descriptor_table.csv
  final_shortlisted_compositions.csv

results/
  benchmark_metrics.csv
  learning_curve_summary.csv

splits/
  learning_curve_splits_seed42.json

notebooks/
  HEA_reproducible_clean.ipynb
```

The `data/`, `results/`, and `splits/` directories contain output files after the workflow is executed. Processed tables may also be placed there directly when raw DFT outputs are not distributed with the repository.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

The raw DFT output directory is expected to contain one folder per composition. Each composition folder may contain several `run_*` subdirectories with `data.json` files and optional geometry files.

Example layout:

```text
HEA_results_19_metals/
  AgCoCrCuFe/
    run_001/
      data.json
      geometry.xyz
    run_002/
      data.json
      geometry.xyz
```

To export the processed composition tables, run:

```bash
python scripts/export_composition_tables.py --base-path HEA_results_19_metals --out-dir data
```

This creates:

```text
data/dft_calculated_alloys.csv
data/descriptor_table.csv
```

## Descriptor definition

The screening descriptor is

```text
D = <E_ads(S)> + 2<E_ads(H)>
```

where `<E_ads(S)>` and `<E_ads(H)>` are composition-level averages over the corresponding adsorption-energy samples.

## Linear-quadratic surrogate

The LQ surrogate approximates the teacher-derived descriptor using elemental and pairwise composition terms:

```text
D_LQ = beta_0 + sum_i beta_i x_i + sum_{i<j} beta_ij x_i x_j
```

where `x_i` indicates the presence of element `i` in the alloy composition. Ridge regression is used for fitting. The regularization strength is selected by cross-validation.

To reproduce the LQ learning-curve experiment, run:

```bash
python scripts/run_lq_learning_curve.py --input data/descriptor_table.csv --out-dir results --split-dir splits
```

The script stores the exact split indices in `splits/` and the metrics in `results/`.

## Random seeds and train/test partitions

All stochastic operations are controlled by a fixed base seed. The default seed is `42`. The split indices used by the learning-curve experiments are stored as JSON files and can be reused to reproduce the reported results.

The data partitions are composition-level partitions. All descriptor values associated with the same composition are assigned consistently to the same partition.

## Code and data availability

The repository provides the processed data interfaces, model-training scripts, split files, metrics, and documentation required to reproduce the machine-learning part of the workflow. Raw DFT output files may be omitted when they are too large for direct repository storage. In that case, the processed DFT-derived tables in `data/` serve as the reproducible input for the ML analysis.
