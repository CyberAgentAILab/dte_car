# On Efficient Estimation of Distributional Treatment Effects under Covariate-Adaptive Randomization

This repository contains the implementation code for reproducing results from the paper ["On Efficient Estimation of Distributional Treatment Effects under Covariate-Adaptive Randomization,"](https://arxiv.org/abs/2506.05945) (ICML 2025) by Undral Byambadalai, Tomu Hirata, Tatsushi Oka, and Shota Yasui. 

## Simulation with Python

### Environment
- Python >= 3.12
- uv >= 0.7.0

### Installation

```bash
# Clone the repository
git clone https://github.com/CyberAgentAILab/dte_car.git
cd dte_car

# Setup Python environment
# See https://docs.astral.sh/uv/ for other installation
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv python pin 3.12
uv venv
source .venv/bin/activate

# For running DTE estimation
uv sync
```

### Run Simulation
You can run the simulation using [simulation.py](./simulation.py). The available arguments for this script are described below.

Arguments:
  - n: Specifies the number of samples for each simulation iteration. This is a required argument and must be a positive integer.
  - iterations: The number of simulation iterations.
  - discrete: Indicates whether the Data Generating Process (DGP) is discrete outcome or continuous outcome. By default, the DGP is continuous. Use this flag to switch to discrete mode.
  - discrete_covariates: Indicates whether the Data Generating Process (DGP) is discrete covariates or not. By default, the DGP uses continuous covariates. Use this flag to switch to discrete.

Example:
  - For 1000 samples with a continuous DGP:
    `--n 1000`
  - For 5000 samples with a discrete DGP:
    `--n 5000 --discrete true`

```bash
python simulation.py --n 5000 --discrete true
```

## Empirical Application with R

### Dependencies
Necessary R packages and their versions: R version 4.3.1,  `knitr_1.49` `RColorBrewer_1.1-3` `xgboost_1.7.5.1`    `ggplot2_3.5.1`      `dplyr_1.1.2`  `readr_2.1.4` `haven_2.5.3`

### Data preparation
To replicate the reanalysis of the field experiment run by [Attanasio et al. (2015)](https://www.aeaweb.org/articles?id=10.1257/app.20130489), download the original dataset at [openICPSR](https://www.openicpsr.org/openicpsr/project/113597/version/V1/view).  
Save the whole folder in the same directory and run [data_preprocess.R](./data_preprocess.R) to prepare the dataset and save it to a CSV file called `microcredit.csv`.

### Data analysis
- Run [empirical_application_microcredit.R](./empirical_application_microcredit.R) to calculate distributional treatment effects and save the results as figures (Figure 3 in the paper)

- [functions.R](./functions.R) contains functions needed to calculate the effects.


## Contributors

- Undral Byambadalai (undral_byambadalai@cyberagent.co.jp)
- Tomu Hirata (tomu.hirata@databricks.com)
