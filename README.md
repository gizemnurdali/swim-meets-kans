# SWIM Meets KANs

Data-driven Sampling for Kolmogorov-Arnold Network Training Without Gradient Descent.

## Overview

This repository unifies KAN, SWIM, and HKAN implementations and includes a new proposed framework for the SWIM-Guided KAN (SG-KAN) algorithm:

- **data**: Datasets from the HKAN paper (5 synthetic test functions and 18 real-world benchmarks), plus saved Optuna search results and 50-seed run outputs
- **hkan**: Original Hierarchical Kolmogorov-Arnold Networks (HKAN) source code
- **sgkan**: New proposed algorithm (SWIM-Guided KAN)
- **pykan**: Original KAN implementation by Liu et al.
- **swimnetworks-paper**: Original SWIM networks paper implementation

## Motivation

The project explores combining SWIM's data-driven methodology with KAN's network. The goal is to leverage the strengths of both approaches for improved network modeling and analysis.

## Key resources to look at

**hkan/**: Original Hierarchical Kolmogorov-Arnold Networks (HKAN) source code
- `hkan.py`: original HKAN model implementation
- `hkan_sanity_check.ipynb`: sanity check and preliminary analysis of the official HKAN implementation (depth vs. expressivity, center selection behavior)
- `tutorial.ipynb`: HKAN walkthrough and usage guide

**swimnetworks-paper/**:
- `swim_sanity_check.ipynb`: sanity check of the official SWIM implementation

**sgkan/**: new proposed algorithm
- `sgkan_model.py`: reusable SG-KAN implementation (SWIM pair sampling, local point collection, edge kernel construction, edge/neuron-level ridge regression)
- `sgkan_notebook.ipynb`: original notebook where the algorithm was built and prototyped from scratch, step by step; `sgkan_model.py` is the script version for reuse in the benchmark notebooks
- `sgkan_parameter_search.py`: Optuna hyperparameter search and tuning for SG-KAN

**Root-level scripts**:
- `datasets.py`: loads the synthetic TF and real-world datasets from `data/`
- `evaluation_metrics.py`: shared MAE/RMSE evaluation utilities used by all three models
- `swim.py`: core SWIM pair-sampling implementation (candidate pair sampling, SWIM probabilities, pair selection)
- `utils.py`: build/fit utilities for SGKAN, KAN and HKAN, including the Optuna hyperparameter search for KAN
- `visualize.py`: plotting helpers for inspecting HKAN/SG-KAN block functions and per-layer R²

**Root-level notebooks**:
- `kan_parameter_optimization.ipynb`: Optuna grid search over KAN's architecture search space for each test function (TF1–TF5), selecting the winning width configuration used throughout the rest of the benchmarking
- `sgkan_kan_hkan_benchmarking_evaluation.ipynb`: main benchmarking notebook, covering (1) HKAN reproduction against the paper's reported hyperparameters and results, (2) 50-seed run experiments for SG-KAN, HKAN, and KAN (RMSE, fit time, and inference time, with median/IQR), and (3) layer-width sensitivity analysis comparing SG-KAN (SWIM vs. random pair selection), HKAN, and KAN across TF1–TF5

**Analysis / figure notebooks** (standalone, used to generate specific thesis figures from already-computed results):
- `edge_comparison_tf3.ipynb`: fits SG-KAN(SWIM) and HKAN at their best TF3 configuration, then plots one fitted edge function from each side by side over the same input range, shading the SG-KAN edge's actual `[lo, hi]` fitting interval. Also builds a per-edge coverage/conditioning summary table (interval coverage fraction, local sample count, local-point-collection method, and kernel-matrix condition number) for every edge in the first layer.
- `all_models_stability.ipynb`: loads the 50-run CSV results for SG-KAN(SWIM) and SG-KAN(Random) across all five test functions, computes summary statistics (median, IQR, std, skewness, min/max) per split, tallies how often each variant wins on median vs. on stability, and produces side-by-side boxplots of the RMSE distributions for each test function.

## Contributions

This repository combines and validates implementations from multiple foundational papers:
- Validation and sanity checks of HKAN and SWIM source codes
- New proposed algorithm (SG-KAN) combining SWIM's data-driven approach with KAN architectures

## Getting Started

See individual folders for specific implementation details and usage instructions.