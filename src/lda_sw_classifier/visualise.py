import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lda_sw_classifier import lib

ALL_POSSIBLE: List[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]

CM = plt.get_cmap("Spectral")

NUM_COLORS = 20


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments including config path, id file, output directory, and plot type.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c", type=Path, required=True, help="Path to the config file.")
    parser.add_argument("--id-file", "-i", type=Path, required=True, help="Path to the file with BMRB ids.")
    parser.add_argument("--output-dir", "-d", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument("--plot-type", "-t", type=str, choices=["protein_bar", "amino_bar"], required=True, help="Plot type.")
    return parser.parse_args()


def build_param_str(parameters: Dict[str, Any]) -> str:
    """
    Construct a string representation of parameters for use in filenames.

    Args:
        parameters (Dict[str, Any]): Dictionary of parameters.

    Returns:
        str: Concatenated parameter string with '.' replaced by '_'.
    """
    keys = [
        "algo", "n_projections", "seed", "k", "device", "with_bins",
        "sample_method", "vote_type", "scale", "eps", "kappa1", "kappa2", "note"
    ]
    return "_".join(str(parameters[k]) for k in keys).replace(".", "_")


def get_output_paths(base_dir: Path) -> Tuple[Path, Path]:
    """
    Create and return directories for labels and plots.

    Args:
        base_dir (Path): Base output directory.

    Returns:
        Tuple[Path, Path]: Paths to the label directory and plot directory.
    """

    plot_dir = base_dir / "plots"
    base_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, plot_dir


def calculate_accuracy_stats(cv_accuracy: Dict[str, Tuple[float]]) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation of cross-validation accuracies.

    Args:
        cv_accuracy (Dict[str, Tuple[float]]): Dictionary of cross-validation accuracies.

    Returns:
        Tuple[float, float]: Mean and standard deviation of accuracies.
    """
    accuracies = [float(v[0]) for v in cv_accuracy.values()]
    return np.mean(accuracies), np.std(accuracies)


def get_label_file_path(lab_dir: Path, bmrb_id: str, algo: str, param_str: str) -> Path:
    """
    Determine the filepath for label CSV based on algorithm and parameters.

    Args:
        lab_dir (Path): Label directory path.
        bmrb_id (str): BMRB ID.
        algo (str): Algorithm name.
        param_str (str): Parameter string.

    Returns:
        Path: Path to the label CSV file.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    if algo in {"filtered", "filtered_bootstrap"}:
        return lab_dir / f"combined_labels_{bmrb_id}_{param_str}.csv"
    elif algo in {"lda", "lda_bootstrap"}:
        return lab_dir / f"labels_{bmrb_id}_lda.csv"
    elif algo == "sw":
        return lab_dir / f"labels_sw_{param_str}.csv"
    else:
        raise ValueError(f"Unsupported algorithm '{algo}' for label file path.")


def load_label_data(bmrb_ids: List[str], lab_dir: Path, algo: str, param_str: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load real and predicted labels for a list of BMRB IDs.

    Args:
        bmrb_ids (List[str]): List of BMRB IDs.
        lab_dir (Path): Label directory.
        algo (str): Algorithm name.
        param_str (str): Parameter string.

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, List[str]]]: Two dictionaries mapping BMRB ID to real and predicted labels respectively.
    """
    label_check = {}
    label_cv = {}
    for bmrb_id in bmrb_ids:
        file_path = get_label_file_path(lab_dir, bmrb_id, algo, param_str)
        df = pd.read_csv(file_path)
        label_check[bmrb_id] = df['real labels'].tolist()
        label_cv[bmrb_id] = df['classified as'].tolist()
    return label_check, label_cv


def protein_bar_plot(parameters, label_check, cv_accuracy, output_dir):
    """
    Plot stacked bar chart showing accuracy contribution per amino acid for each protein.

    Args:
        parameters (dict): Parameters dictionary.
        label_check (dict): Dictionary of real labels per protein.
        cv_accuracy (dict): Cross-validation accuracies.
        output_dir (Path): Directory to save the plot.
    """
    mean_accuracy, std_accuracy = calculate_accuracy_stats(cv_accuracy)

    prot = list(label_check.keys())
    weight = [[x.count(lab)/len(x) for lab in ALL_POSSIBLE] for _, x in label_check.items()]
    acc = [float(x[0]) for x in cv_accuracy.values()]
    wxacc = [[x*l for l in k] for k, x in zip(weight, acc)]

    sorted_indices = np.argsort([int(lab) for lab in prot])
    sorted_prot = [prot[i] for i in sorted_indices]
    sorted_wxacc = [wxacc[i] for i in sorted_indices]

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=[CM(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for i, aa in enumerate(ALL_POSSIBLE):
        bar_vals = [w[i] for w in sorted_wxacc]
        bottoms = [sum(w[:i]) for w in sorted_wxacc] if i > 0 else None
        ax.bar(sorted_prot, bar_vals, bottom=bottoms, label=aa, edgecolor="black", linewidth=0.5)

    for i, total in enumerate([sum(w) for w in sorted_wxacc]):
        ax.text(sorted_prot[i], total + 0.01, round(total, 3), ha='center', weight='bold', color='black')

    ax.set_ylabel('Accuracy')
    plt.yticks([0.05 * i for i in range(1, 21)])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    algo = parameters["algo"]
    title_map = {
        "full": "SW Classifier combined",
        "filtered": "SW Classifier combined with filtration",
        "sw": "SW Classifier",
        "lda": "LDA",
        "lda_bootstrap": "LDA with bootstrap",
        "full_bootstrap": "SW Classifier combined with LDA bootstrap",
        "filtered_bootstrap": "SW Classifier combined with filtration and LDA bootstrap"
    }
    title_prefix = title_map.get(algo, "Unknown")
    k_str = f"k = {parameters['k']}, " if parameters["k"] else ""
    title = (
        f"{title_prefix} no. projections = {parameters['n_projections']}, "
        f"{k_str}average accuracy = {mean_accuracy:.3f} ± {std_accuracy:.3f}"
    )
    plt.title(title, size=16)

    fig.set_size_inches(16, 6)
    param_str = build_param_str(parameters)
    fname = f"validation_protein_bar_plot_{param_str if 'lda' not in algo else algo}.png"
    plt.savefig(Path(output_dir) / fname)
    plt.show()


def amino_bar_plot(parameters, label_check, cv_dict, cv_accuracy, output_dir):
    """
    Plot recall and precision bar charts for amino acids based on predicted vs. real labels.

    Args:
        parameters (dict): Parameters dictionary.
        label_check (dict): Dictionary of real labels per protein.
        cv_dict (dict): Dictionary of predicted labels per protein.
        cv_accuracy (dict): Cross-validation accuracies.
        output_dir (Path): Directory to save the plot.
    """
    mean_accuracy, std_accuracy = calculate_accuracy_stats(cv_accuracy)
    amino_ok, amino_all, amino_pred = {}, {}, {}

    for amino in ALL_POSSIBLE:
        for prot, labels in label_check.items():
            test_labels = cv_dict[prot]
            for lab1, lab2 in zip(labels, test_labels):
                if lab1 == amino:
                    amino_all[amino] = amino_all.get(amino, 0) + 1
                    if lab2 == amino:
                        amino_ok[amino] = amino_ok.get(amino, 0) + 1
                if lab2 == amino:
                    amino_pred[amino] = amino_pred.get(amino, 0) + 1

    amino_rec = {aa: amino_ok.get(aa, 0)/amino_all.get(aa, 1) for aa in ALL_POSSIBLE}
    amino_prec = {aa: amino_ok.get(aa, 0)/amino_pred.get(aa, 1) for aa in ALL_POSSIBLE}

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    key1 = list(amino_rec.keys())
    val1 = list(amino_rec.values())
    ax1.set_prop_cycle(color=[CM(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    ax1.bar(key1, val1, edgecolor = "black", linewidth = 0.5, color=[CM(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for i, total in enumerate(val1):
        ax1.text(key1[i], total + 0.01, f"{amino_rec[key1[i]]:.3f}",
                ha='center', weight='bold', color='black')

    key2 = list(amino_prec.keys())
    val2 = list(amino_prec.values())
    ax2.set_prop_cycle(color=[CM(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    ax2.bar(key2, val2, edgecolor = "black", linewidth = 0.5, color=[CM(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for i, total in enumerate(val2):
        ax2.text(key2[i], total + 0.01, f"{amino_prec[key2[i]]:.3f}",
                ha='center', weight='bold', color='black')

    ax1.set_ylabel('Recall per aminoacid')
    recall_values = np.array([float(v) for _, v in amino_rec.items()])
    avg_recall = np.mean(recall_values)
    std_recall = np.std(recall_values)
    ax1.set_title(f"Average recall for aminoacids: {avg_recall:.3f}, Std: {std_recall:.3f}")

    ax2.set_ylabel('Precision per aminoacid')
    prec_values = np.array([float(v) for _, v in amino_prec.items()])
    avg_prec = np.mean(prec_values)
    std_prec = np.std(prec_values)
    ax2.set_title(f"Average precision for aminoacids: {avg_prec:.3f}, Std: {std_prec:.3f}")

    algo = parameters["algo"]
    title_prefix = {
        "full": "SW Classifier combined",
        "filtered": "SW Classifier combined with filtration",
        "sw": "SW Classifier",
        "lda": "LDA",
        "lda_bootstrap": "LDA with bootstrap",
        "full_bootstrap": "SW Classifier combined with LDA bootstrap",
        "filtered_bootstrap": "SW Classifier combined with filtration and LDA bootstrap"
    }.get(algo, "Unknown")

    k_str = f"k = {parameters['k']}, " if parameters["k"] else ""
    sup_title = (
        f"{title_prefix} no. projections = {parameters['n_projections']}, "
        f"{k_str}average accuracy = {mean_accuracy:.3f} ± {std_accuracy:.3f}"
    )
    plt.suptitle(sup_title, size=16)
    plt.yticks([0.05 * i for i in range(1, 21)])

    param_str = build_param_str(parameters)
    fname = f"validation_amino_bar_plot_{param_str if 'lda' not in algo else algo}.png"
    plt.savefig(Path(output_dir) / fname)
    plt.show()


def main() -> None:
    """
    Main function to parse arguments, load configuration and data,
    and generate the specified plot type.
    """
    warnings.filterwarnings("ignore")
    args = parse_arguments()

    config = lib.load_config(args.config_path)
    parameters = lib.prepare_parameters(config)
    algo = parameters["algo"]
    param_str = build_param_str(parameters)

    lab_dir, plot_dir = get_output_paths(args.output_dir)

    acc_filename = (
        f"accuracy_validation_{algo}.csv"
        if algo in {"lda", "lda_bootstrap"}
        else f"accuracy_validation_combined_{param_str}.csv"
    )
    accuracy_file = lab_dir / acc_filename
    cv_accuracy = pd.read_csv(accuracy_file, index_col=0).to_dict()

    bmrb_ids = lib.read_bmrb_id(args.id_file)
    label_check, label_cv = load_label_data(bmrb_ids, lab_dir, algo, param_str)

    if args.plot_type == "protein_bar":
        protein_bar_plot(parameters, label_check, cv_accuracy, plot_dir)
    elif args.plot_type == "amino_bar":
        amino_bar_plot(parameters, label_check, label_cv, cv_accuracy, plot_dir)


if __name__ == "__main__":
    main()
