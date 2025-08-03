import argparse
import time
import warnings
from pathlib import Path

from lda_sw_classifier import lib


PARAMS_AMINO = {
    "ALA": 0.3, "ARG": 0.3, "ASN": 0.2, "ASP": 0.3, "CYS": 0.22, "GLN": 0.3, "GLU": 0.9,
    "GLY": 0.5, "HIS": 0.3, "ILE": 0.3, "LEU": 0.4, "LYS": 0.5, "MET": 0.3, "PHE": 0.2,
    "PRO": 0.7, "SER": 0.5, "THR": 0.5, "TRP": 0.2, "TYR": 0.2, "VAL": 0.2
}


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments including paths for config, ID file,
        Excel protein file, FASTA file, and output directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        "-c",
        type=str,
        required=True,
        help="Path to the config file."
    )
    parser.add_argument(
        "--id-file",
        "-i",
        type=str,
        required=True,
        help="Path to the file with BMRB ids."
    )
    parser.add_argument(
        "--excel-protein",
        "-p",
        type=str,
        required=True,
        help="Path to the Excel file with protein to analyse."
    )
    parser.add_argument(
        "--fasta",
        "-f",
        type=str,
        required=True,
        help="Path to the FASTA file of the protein to analyse."
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        required=True,
        help="Path to the output directory."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to perform classification of a single protein using LDA and optionally
    a Sliced Wasserstein or combined approach.
    """
    warnings.filterwarnings("ignore")
    args = parse_arguments()

    config_path = Path(args.config_path)
    id_file = Path(args.id_file)
    excel_protein = Path(args.excel_protein)
    fasta_file = Path(args.fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    config = lib.load_config(str(config_path))
    parameters = lib.prepare_parameters(config)
    bmrb_id = lib.read_bmrb_id(str(id_file))

    param_str = "_".join([str(v) for v in parameters.values()]).replace(".", "_")
    t1 = time.time()

    content_lda = lib.lda_classification_classify(
        parameters, str(excel_protein), bmrb_id, str(fasta_file), str(output_dir)
    )

    if parameters["algo"] not in ["lda", "lda_bootstrap"]:
        train, test, _ = lib.prepare_data(
            parameters["algo"], content_lda, parameters["sample_method"],
            parameters["seed"], parameters["eps"]
        )
        content_sw = lib.validate(
            1, train, test, parameters, PARAMS_AMINO, param_str, str(output_dir)
        )
        if parameters["algo"] != "sw":
            content_combined = lib.combine_methods(
                str(output_dir), content_lda, content_sw, param_str
            )
            print(f"CONTENT combined: {content_combined}")

    t2 = time.time()
    print(f"\nClassification took {t2 - t1:.2f} s.")


if __name__ == "__main__":
    main()
