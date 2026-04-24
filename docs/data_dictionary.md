# Data dictionary

## data/dft_calculated_alloys.csv

```text
composition
n_runs
E_ads_S_mean
E_ads_S_std
E_ads_H_mean
E_ads_H_std
D
```

## data/descriptor_table.csv

```text
composition
D
E_ads_S_mean
E_ads_H_mean
x_<element>
```

The `x_<element>` columns are binary composition indicators.

## data/final_shortlisted_compositions.csv

```text
rank
composition
predicted_D
selection_stage
reason_for_selection
```

## results/benchmark_metrics.csv

```text
model
train_size
fold
seed
MAE
R2
kendall_tau
```

## splits/learning_curve_splits_seed42.json

```text
train_size
fold
seed
train_indices
test_indices
```
