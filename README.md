# Code for Reproducibility

This repository contains all the code used to generate the figures in *...*.

## Util Files
- `binary_links.py` contains a class that implements the derivatives of the inverse link (quantile) functions as well as the that of the observed Fisher Information.
  - A few notes about the class is also contained in the associated .md file of the same name.
- `data_generating_utils.py' contains the function which generates the data for all the simulations.
- `ddc_utils.py` contains the primary util file, containing helpers for all the computation used to compute the data defect correlation, realized efficiency, etc...

## Figures 

The figures are split into three separate folders: one for purely Section 4.1, one for purely Section 4.2, and the rest.
- For the folder for only Section 4.1, each sampling scheme was split into its own notebook.
  - The generated data is then stored in separate pickle files.
  - Afterwards, all the data is plotted altogether.
- For the folder for only Section 4.2, each combination of (link function, beta) has a unique notebook, from which the data is generated.
  - The generated data is then stored in separate pickle files.
  - Afterwards, all the data is aggregated into a singular place in a single notebook.

