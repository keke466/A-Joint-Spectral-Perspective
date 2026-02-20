# Code for "Average-Case Analysis with Anisotropic Initialization"

This repository contains the Python code to reproduce the numerical experiments in the paper.

## Requirements
- numpy
- scipy
- matplotlib

Install with: `pip install numpy scipy matplotlib`

## Files
- `generate_table_gd_nesterov.py`: runs 20 trials for each initialization and prints LaTeX table.
- `plot_gd_nesterov.py`: generates convergence curves (Figure 1).

## Usage
Run `python generate_table_gd_nesterov.py` to get the table data.
Run `python plot_gd_nesterov.py` to generate the figure.
