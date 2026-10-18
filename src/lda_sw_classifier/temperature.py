
TEMPERATURE_COEFFICIENTS = {
    "CA": {
        "A": -2.2, "C": -0.9, "D": 2.8, "E": 0.9,
        "F": -4.7, "G": 3.3, "H": 7.8, "I": -2.0,
        "K": -0.8, "L": 1.7, "M": 4.1, "N": 2.8,
        "P": 1.1, "Q": 2.3, "R": -1.4, "S": -1.7,
        "T": 0.0, "V": -2.8, "W": -2.7, "Y": -5.0,
    },
    "CB": {
        "A": 4.7, "C": 1.3, "D": 6.5, "E": 4.6,
        "F": 2.4, "G": None, "H": 15.5, "I": 4.6,
        "K": 2.4, "L": 4.9, "M": 9.4, "N": 5.1,
        "P": -0.2, "Q": 3.6, "R": 3.5, "S": 4.4,
        "T": 2.2, "V": 2.5, "W": 3.1, "Y": 2.9,
    },
    "CO": {
        "A": -7.1, "C": -2.6, "D": -4.8, "E": -4.9,
        "F": -6.9, "G": -3.2, "H": 3.1, "I": -8.7,
        "K": -7.1, "L": -8.2, "M": -8.2, "N": -6.1,
        "P": -4.0, "Q": -5.7, "R": -6.9, "S": -4.7,
        "T": -5.2, "V": -8.1, "W": -7.9, "Y": -7.7,
    },
    "N": {
        "A": -5.3, "C": -8.2, "D": -3.9, "E": -3.7,
        "F": -11.2, "G": -6.2, "H": 3.3, "I": -12.7,
        "K": -7.6, "L": -2.9, "M": -6.2, "N": -3.3,
        "P": None, "Q": -6.5, "R": -5.3, "S": -3.8,
        "T": -6.7, "V": -14.2, "W": -10.1, "Y": -12.0,
    },
    "HN": {
        "A": -9.0, "C": -7.0, "D": -6.2, "E": -6.5,
        "F": -7.5, "G": -9.1, "H": -7.8, "I": -7.8,
        "K": -7.5, "L": -7.5, "M": -7.1, "N": -7.0,
        "P": None, "Q": -7.2, "R": -7.1, "S": -7.6,
        "T": -7.3, "V": -7.6, "W": -7.8, "Y": -7.7,
    },
    "HA": {
        "A": 0.7, "C": 0.0, "D": -0.1, "E": 0.3,
        "F": 0.4, "G": 0.0, "H": 0.4, "I": 0.4,
        "K": 0.4, "L": 0.1, "M": -0.5, "N": -2.9,
        "P": 0.0, "Q": 0.3, "R": 0.4, "S": 0.1,
        "T": 0.0, "V": 0.5, "W": 0.4, "Y": 0.5,
    },
}

import pandas as pd


def _clean_nmrstar_value(value):
    """Convert missing NMR-STAR markers to None."""
    if value is None:
        return None

    value = str(value).strip()

    if value in {"", ".", "?"}:
        return None

    return value


def get_temperature_for_shift_list(ent, shift_frame):
    """
    Return the experimental temperature in Kelvin for a
    chemical-shift saveframe, or None if it cannot be established.
    """
    shift_list_ids = shift_frame.get_tag(
        "_Assigned_chem_shift_list.Sample_condition_list_ID"
    )

    if not shift_list_ids:
        return None

    condition_id = _clean_nmrstar_value(shift_list_ids[0])

    if condition_id is None:
        return None

    for loop in ent.get_loops_by_category("Sample_condition_variable"):
        condition_ids = loop.get_tag(
            "_Sample_condition_variable.Sample_condition_list_ID"
        )
        types = loop.get_tag(
            "_Sample_condition_variable.Type"
        )
        values = loop.get_tag(
            "_Sample_condition_variable.Val"
        )
        units = loop.get_tag(
            "_Sample_condition_variable.Val_units"
        )

        for cid, variable_type, value, unit in zip(
            condition_ids, types, values, units
        ):
            if _clean_nmrstar_value(cid) != condition_id:
                continue

            if _clean_nmrstar_value(variable_type) is None:
                continue

            if "temp" not in variable_type.lower():
                continue

            value = _clean_nmrstar_value(value)
            unit = _clean_nmrstar_value(unit)

            if value is None or unit is None:
                continue

            try:
                temperature = float(value)
            except ValueError:
                continue

            unit_normalized = unit.strip().lower()

            if unit_normalized in {"k", "kelvin", "kelvins"}:
                return temperature

            if unit_normalized in {
                "c", "°c", "deg c", "degree celsius",
                "degrees celsius", "celsius",
            }:
                return temperature + 273.15

    return None


AMINO_ACID_3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}



def correct_temperature_shifts(
    df,
    reference_temperature=298.0,
):
    """
    Normalize chemical shifts to a reference temperature.

    Temperature coefficients are in ppb/K; chemical shifts are in ppm.
    Rows without a known temperature or coefficient remain unchanged.
    """
    corrected = df.copy()

    print("In correct shifts")

    if "temperature_K" not in corrected.columns:
        raise ValueError(
            "DataFrame must contain a 'temperature_K' column."
        )

    if "amino" not in corrected.columns:
        raise ValueError(
            "DataFrame must contain an 'amino' column."
        )

    temperature = pd.to_numeric(
        corrected["temperature_K"], errors="coerce"
    )
    delta_t = temperature - reference_temperature

    amino = (
        corrected["amino"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(AMINO_ACID_3_TO_1)
    )

    for feature, coefficient_map in TEMPERATURE_COEFFICIENTS.items():
        if feature not in corrected.columns:
            continue

        beta = pd.to_numeric(
            amino.map(coefficient_map), errors="coerce"
        )

        shift = pd.to_numeric(
            corrected[feature], errors="coerce"
        )

        valid = (
            temperature.notna()
            & beta.notna()
            & shift.notna()
        )

        print(
            f"[DEBUG] feature={feature}, "
            f"rows={len(corrected)}, "
            f"valid_rows={int(valid.sum())}, "
            f"temperature_available={int(temperature.notna().sum())}, "
            f"amino_acid_recognized={int(beta.notna().sum())}, "
            f"shift_available={int(shift.notna().sum())}",
            flush=True,
        )

        corrected_values = (
            shift.loc[valid]
            - beta.loc[valid] * delta_t.loc[valid] / 1000.0
        )

        for idx in corrected.index[valid]:
            print(
                f"[Temperature correction] "
                f"row={idx}, "
                f"amino={amino.loc[idx]}, "
                f"nucleus={feature}, "
                f"before={shift.loc[idx]:.4f} ppm, "
                f"after={corrected_values.loc[idx]:.4f} ppm, "
                f"temperature_K={temperature.loc[idx]:.2f}, "
                f"reference_temperature={reference_temperature:.2f} K"
            )

        corrected.loc[valid, feature] = corrected_values

    return corrected
