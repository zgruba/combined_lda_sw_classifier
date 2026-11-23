# Combined LDA-SW classifier

The implementation of combined LDA-SW classifier. We present a novel approach that integrates Linear Discriminant Analysis (LDA) with a classification strategy based on the Sliced-Wasserstein (SW) distance, grounded in optimal transport theory. The two methods are combined using a decision tree to improve the robustness and accuracy of amino acid assignment in high-dimensional NMR spectra of IDPs.

The implementation of LDA follows the methodology introduced in Romero et al.[1], available at https://github.com/gugumatz/LDA-for-mapping-IDPs.

One adjustment to the SW-based approach, aimed at improving classification accuracy, is the use of Wasserstein transport with auxiliary points, as proposed by Domżał et al. [2].

[1] Romero, J. A., Putko, P., Urbańczyk, M., Kazimierczuk, K. & Zawadzka-Kazimierczuk, A. Linear discriminant analysis reveals hidden patterns in NMR chemical shifts of intrinsically disordered proteins. PLoS Comput. Biol. 18, e1010258 (2022). DOI: [10.1371/journal.pcbi.1010258](https://doi.org/10.1371/journal.pcbi.1010258)

[2] Domżał, B., Nawrocka, E. K., Gołowicz, D., Ciach, M. A., Miasojedow, B., Kazimierczuk, K. & Gambin, A. Magnetstein: An open-source tool for quantitative NMR mixture analysis robust to low resolution, distorted lineshapes, and peak shifts. Anal. Chem. 95, 7532–7541 (2023). DOI: [10.1021/acs.analchem.3c03594](https://doi.org/10.1021/acs.analchem.3c03594).


## Repository Overview

The code is structured into the following key components:

- `src/lda_sw_classifier/` — source code for the classifier
  - `classify.py`
  - `validate.py`
  - `visualise.py`
  - `lib.py`
- `requirements.txt` — Python dependencies
- `pyproject.toml` — project build configuration
- `README.md` — project overview and instructions
- `LICENSE` — licensing information

## Installation

To use the software in this repository, make sure you have **Python 3.12.7** installed on your system.

### 1. Clone the repository

Open your terminal and run:

```bash
git clone <repository_url>
```

### 2. Navigate to the project directory

```bash
cd combined_lda_sw_classifier/
```

### 3. Install the package

Use pip to install the package:

```bash
pip install .
```

## Software Package Usage

This paragraph outlines the intended usage of the software package developed as part of this research.

### 1. Classification

The package includes a script for performing classification analyses. It accepts a configuration file, a text file containing BMRB IDs, an Excel file with protein chemical shifts, and an output directory.

**Command-line usage:**

```bash
lda_sw_classify -c [config_file] -i [bmrbID_file] -p [excel_protein] -d [output_directory]
```
**Output:**

The output directory will contain two .csv files: one with predicted labels and another with the classification scores (probabilities) used to make those predictions.

### 2. Validation

A separate script is provided for validating classification results. It takes a configuration file and a text file containing BMRB IDs as input.

**Command-line usage:**

```bash
lda_sw_validate -c [config_file] -i [bmrbID_file] -d [output_directory]
```
**Output:**

The output directory will contain two .csv files for each protein from the BMRB ID list: one with predicted labels and another with the classification scores (probabilities) used to generate those predictions. Additionally, summary .csv files with LOO cross-validation accuracy and processing times will be included.

### 3. Visualisation of Validation Results

This script generates plots to facilitate interpretation of validation results. It supports multiple visualisation modes as specified via the `-o` option.

**Command-line usage:**

```bash
lda_sw_visualise -c [config_file] -i [bmrbID_file] -o [plot_type] -d [output_directory]
```
**Output:**

The output directory will include a plots/ subdirectory containing .png files with the corresponding classification plots.

### Configuration File Format

The configuration file follows the standard TOML format. A typical example is shown below:

```toml
[parameters]
algo = "filtered"       # Options: lda, lda_bootstrap, sw, filtered, filtered_bootstrap
n_projections = 4000
seed = 42
k = 0
device = "cpu"
with_bins = true
sample_method = "BEST"  # Options: RAND, CENT
vote_type = "SPECIAL"   # Options: NORMAL
scale = true
eps = 0.75
kappa1 = 2.1
kappa2 = 2.1
note = "output_note"
```
If you only want to run LDA, set algo = "lda". In this case, all SW-related parameters (n_projections, seed, k, with_bins, sample_method, vote_type, scale, eps, kappas, and note) should be set to None or kept as they are for SW. Do not leave them empty.

The parameter sample_method set to "BEST" indicates that a density-based selection approach is employed, whereas "RAND" corresponds to random sampling, and "CENT" denotes the utilisation of class centroids. The vote_type parameter set to "SPECIAL" signifies the incorporation of a projections cutoff, while "NORMAL" indicates the absence of such a cutoff. The parameter k defaults to 0, but you may set it to any positive integer. When set, SW classification will use the top-k source points (those with the highest transported mass) for each target point in the voting procedure.

### Combined methods with W (multidimensional Wasserstein)

The `configs/` folder holds ready-to-use configs for four combined methods. Each one replaces SW with the full Wasserstein transport (wnet) and the decision tree with a combination rule:

| Config | Method | Accuracy, 263 proteins (5D): no correction / T+pH standard / T+pH test |
|---|---|---|
| `configs/lda_w_mix.toml` | LDA+W, mix | 88.96 / 88.91 / 88.96 % |
| `configs/lda_w_assign.toml` | LDA+W, assign | 92.14 / 92.25 / 92.35 % |
| `configs/et_w_mix_open.toml` | ET+LDA+W, mix, open | 91.94 / 91.93 / 92.16 % |
| `configs/et_w_assign_open.toml` | ET+LDA+W, assign, open | 94.28 / 94.43 / 94.48 % |

```bash
lda_sw_validate -c configs/et_w_assign_open.toml -i [bmrbID_file] -d [output_directory]
lda_sw_classify -c configs/et_w_assign_open.toml -i [bmrbID_file] -p [excel_protein] -f [fasta_file] -d [output_directory]
```

The keys that select these methods (all optional; without them the thesis pipeline runs unchanged):

- `transport = "w"` with `sample_method = "ALL"`: W instead of SW. The source is every training point of the amino acids left in the composition, each class weighted by its share of the composition. `w_metric` (`L1`, `L2`, `LINF`) is the cost, and `kappa1`/`kappa2` are the costs of the auxiliary (trash) points.
- `classifier = "et_lda"`: ExtraTrees for residues with at least 2 of CA/HB/CB measured, LDA for the rest.
- `et_filter = "certain"` (open mode): only residues with probability exactly 1 are fixed. With `"all"` (the default), every ExtraTrees residue is fixed.
- `combine`:
  - `"tree"` (default): the thesis decision tree.
  - `"mix"`: the probability is `combine_weight` · P_classifier + (1 − `combine_weight`) · P_W.
  - `"assign"`: residues are assigned to composition slots (`scipy.optimize.linear_sum_assignment`). The score is `combine_weight` · log P_classifier + (1 − `combine_weight`) · log(P_W + `combine_floor`), with `combine_slack` × composition slots per amino acid.

Temperature and pH correction (`temperature_correction`, `ph_correction`) use `correction_reference`:

- `"standard"`: all proteins are corrected to `reference_temperature`/`reference_ph`.
- `"test"`: in `lda_sw_validate`, the training proteins are corrected to the temperature and pH of the test protein, which stays as measured. The pKa of the pH reference state is taken at the test protein's temperature and ionic strength.

`lda_sw_classify` corrects only the training data, always to `reference_temperature`/`reference_ph`; the chemical shifts of the classified protein are used as given. The assign rule takes the composition from the FASTA file and scales the slots to the number of residues being classified, so the closer the FASTA matches the residues in the spectrum, the better.

# Citing

If you use tools from this package, please cite:

Kozaryna, Z. Sliced-Wasserstein Optimal Transport as a Method of Enhancing Amino Acid Classification of Intrinsically Disordered Proteins in NMR Data (2025)
