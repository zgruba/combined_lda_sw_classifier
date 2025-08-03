import argparse
import time
import warnings
import pandas as pd

from typing import Any, Dict
from pathlib import Path

from lda_sw_classifier import lib


PARAMS_AMINO: Dict[str, float] = {
    "ALA": 0.3, "ARG": 0.3, "ASN": 0.2, "ASP": 0.3, "CYS": 0.22, "GLN": 0.3, "GLU": 0.9,
    "GLY": 0.5, "HIS": 0.3, "ILE": 0.3, "LEU": 0.4, "LYS": 0.5, "MET": 0.3, "PHE": 0.2,
    "PRO": 0.7, "SER": 0.5, "THR": 0.5, "TRP": 0.2, "TYR": 0.2, "VAL": 0.2
}


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - config_path (str): Path to the configuration file.
            - id_file (str): Path to the file with BMRB IDs.
            - output_dir (str): Path to the output directory.
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
        "--output-dir",
        "-d",
        type=str,
        required=True,
        help="Path to the output directory."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution function for the LDA and Sliced Wasserstein classification pipeline.
    """
    warnings.filterwarnings("ignore")
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    config: Dict[str, Any] = lib.load_config(args.config_path)
    parameters: Dict[str, Any] = lib.prepare_parameters(config)
    bmrb_id: list[str] = lib.read_bmrb_id(args.id_file)

    param_str: str = "_".join([str(v) for v in parameters.values()]).replace(".", "_")

    proteins: Dict[str, Dict[str, Any]] = {}
    cv_accuracy: Dict[str, float] = {}
    times: Dict[str, float] = {}

    for i, protein in enumerate(bmrb_id):
        t1: float = time.time()
        new_param_str: str = f"{protein}_{param_str}"
        proteins[protein] = {}
        print(f"\nConducting LDA for {protein}...")
        content_lda: Dict[str, Any] = lib.lda_classification_validate(parameters, protein, bmrb_id, str(output_dir))
        proteins[protein]["content_lda"] = content_lda

        if parameters["algo"] not in ["lda", "lda_bootstrap"]:
            print(f"\nConducting SW for {protein}...")
            train, test, _ = lib.prepare_data(
                parameters["algo"], content_lda, parameters["sample_method"],
                parameters["seed"], parameters["eps"]
            )
            content_sw: Dict[str, Any] = lib.validate(
                i+1, train, test, parameters, PARAMS_AMINO, new_param_str, str(output_dir)
            )
            proteins[protein]["content_sw"] = content_sw

            if parameters["algo"] != "sw":
                print(f"Combining methods for {protein}...")
                content_combined: Dict[str, Any] = lib.combine_methods(
                    str(output_dir), content_lda, content_sw, new_param_str
                )
                proteins[protein]["content_combined"] = content_combined

        t2: float = time.time()
        print(f"Protein {protein} classification took {t2 - t1:.2f} s.")
        times[protein] = t2 - t1

    df_times: pd.DataFrame = pd.DataFrame(times, index=["Time"])
    df_times.to_csv(output_dir / f"times_{param_str}.csv")

    if parameters["algo"] in ["lda", "lda_bootstrap"]:
        for protein in bmrb_id:
            cv_accuracy[protein] = proteins[protein]["content_lda"]["lda_accuracy"]
        df_accuracy = pd.DataFrame(cv_accuracy)
        df_accuracy.to_csv(output_dir / "accuracy_validation_lda.csv")

    elif parameters["algo"] == "sw":
        for protein in bmrb_id:
            cv_accuracy[protein] = proteins[protein]["content_sw"]["sw_accuracy"]
        df_accuracy = pd.DataFrame(cv_accuracy)
        df_accuracy.to_csv(output_dir / f"accuracy_validation_sw_{param_str}.csv")

    else:
        for protein in bmrb_id:
            cv_accuracy[protein] = proteins[protein]["content_combined"]["combined_accuracy"]
        df_accuracy = pd.DataFrame(cv_accuracy)
        df_accuracy.to_csv(output_dir / f"accuracy_validation_combined_{param_str}.csv")


if __name__ == "__main__":
    main()
