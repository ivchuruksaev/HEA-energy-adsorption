# Reproducibility protocol

## Random seeds

All stochastic operations are controlled by a fixed base seed. The default value is `42`.

The following random number generators are initialized before model fitting:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
```

When CUDA is available, deterministic execution settings are enabled where supported by the installed PyTorch version.

## Data partitions

All partitions are created at the composition level. A composition is assigned either to the training subset or to the test subset. The same composition does not appear in both subsets within the same fold.

For learning-curve experiments, shuffled K-fold partitions are created using the fixed base seed. For each training-set size, the training subset is sampled only from the training fold. The test fold remains unchanged.

The exact split indices are saved in JSON format. This makes the reported learning curves independent of implicit random-state behaviour.

## LQ surrogate

The linear-quadratic surrogate is fitted with ridge regression:

```text
D_LQ = beta_0 + sum_i beta_i x_i + sum_{i<j} beta_ij x_i x_j
```

The linear terms represent average elemental contributions. The quadratic terms represent pairwise corrections associated with the joint presence of two elements in the composition. The physical motivation for these terms comes from the local-environment nature of adsorption on high-entropy alloy surfaces.

## Evaluation

The main ranking metric is Kendall's tau between the LQ-predicted descriptor and the teacher-derived descriptor. This metric is used because the screening workflow depends primarily on preserving the ordering of candidate materials.

The reported metrics are computed over repeated folds and summarized by mean and standard deviation.
