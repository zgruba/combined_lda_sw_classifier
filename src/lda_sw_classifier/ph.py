"""
pH correction of backbone chemical shifts, ported from POTENCI
(Nielsen & Mulder, J. Biomol. NMR 70, 141-165, 2018).

pKa values of titratable sites are estimated from the sequence with a
screened electrostatic model; the protonated fraction of every site then
gives the shift change of the residue (and its neighbours) relative to
the reference pH. Only the pH part is used here, the temperature part
lives in temperature.py.
"""

import hashlib
import os

from datetime import datetime
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.special import erfc

from lda_sw_classifier.temperature import (
    AMINO_ACID_3_TO_1,
    _clean_nmrstar_value,
)


# Constants of the electrostatic model.
GAS_CONSTANT = 8.314472
DIELECTRIC = 79.0
DISTANCE_A = 5.0
DISTANCE_B = 7.5
CUTOFF = 2
N_CYCLES = 5

DEFAULT_IONIC_STRENGTH = 0.1   # M
DEFAULT_TEMPERATURE = 298.0    # K

# Fitted pKa values {(seq, temperature, ionic_strength): pkas}, shared by all runs
# through a file whose name changes whenever this module changes.
PKA_CACHE = {}
PKA_CACHE_FILE = (
    Path(__file__).resolve().parents[2] / "Input" / "bmrb_cache"
    / f"pka_{hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:16]}.pkl"
)

# Intrinsic pKa values; "n" and "c" are the N- and C-terminus.
PK0 = {
    "n": 8.23, "D": 3.86, "E": 4.34, "H": 6.45, "C": 8.49,
    "K": 10.34, "R": 13.9, "Y": 9.76, "c": 3.55,
}

# POTENCI atom names -> classifier feature names.
ATOM_TO_FEATURE = {
    "H": "HN", "N": "N", "C": "CO", "CA": "CA",
    "CB": "CB", "HA": "HA", "HB": "HB",
}

# Columns: residue, atom, shift protonated, shift deprotonated,
# difference, and optionally the difference for residues i-1 and i+1.
PH_SHIFT_TABLE = """
D H  8.55 8.38 -0.17 0.02 -0.03
D HA 4.78 4.61 -0.17 0.01 -0.01
D HB 2.93 2.70 -0.23
D CA 52.9 54.3 1.4 0.0 0.1
D CB 38.0 41.1 3.0
D CG 177.1 180.3 3.2
D C  175.8 176.9 1.1 -0.2 0.4
D N  118.7 120.2 1.5  0.3 0.1
D Np na na 0.1
E H  8.45 8.57  0.12 0.00 0.02
E HA 4.39 4.29 -0.10 0.01 0.00
E HB 2.08 2.02 -0.06
E HG 2.49 2.27 -0.22
E CA 56.0 56.9 1.0 0.0 0.0
E CB 28.5 30.0 1.5
E CG 32.7 36.1 3.5
E CD 179.7 183.8 4.1
E C  176.5 177.0 0.6  0.1 0.1
E N  119.9 120.9 1.0  0.2 0.1
E Np na na 0.1
H H  8.55 8.35 -0.2  -0.01  0.0
H HA 4.75 4.59 -0.2  -0.01 -0.06
H HB 3.25 3.08 -0.17
H HD2 7.30 6.97 -0.33
H HE1 8.60 7.68 -0.92
H CA 55.1 56.7 1.6 -0.1 0.1
H CB 28.9 31.3 2.4
H CG 131.0 135.3 4.2
H CD2 120.3 120.0 -0.3
H CE1 136.6 139.2 2.6
H C 174.8 176.2 1.5  0.0 0.6
H N 117.9 119.7 1.8  0.3 0.5
H Np na na 0.5
H ND1 175.8 231.3 56
H NE2 173.1 181.1 8
C H 8.49 8.49 0.0
C HA 4.56 4.28 -0.28 -0.01 -0.01
C HB 2.97 2.88 -0.09
C CA 58.5 60.6 2.1 0.0 0.1
C CB 28.0 29.7 1.7
C C 175.0 176.9 1.9 -0.4 0.5
C N 118.7 122.2 3.6  0.4 0.6
C Np na na 0.6
Y H  8.16 8.16 0.0
Y HA 4.55 4.49 -0.06
Y HB 3.02 2.94 -0.08
Y HD 7.14 6.97 -0.17
Y HE 6.85 6.57 -0.28
Y CA 58.0 58.2 0.3
Y CB 38.6 38.7 0.1
Y CG 130.5 123.8 -6.7
Y CD 133.3 133.2 -0.1
Y CE 118.4 121.7 3.3
Y CZ 157.0 167.4 10.4
Y C 176.3 176.7 0.4
Y N 120.1 120.7 0.6
K H  8.4  8.4  0.0
K HA 4.34 4.30 -0.04
K HB 1.82 1.78 -0.04
K HG 1.44 1.36 -0.08
K HD 1.68 1.44 -0.24
K HE 3.00 2.60 -0.40
K CA 56.4 56.9 0.4
K CB 32.8 33.2 0.3
K CG 24.7 25.0 0.4
K CD 28.9 33.9 5.0
K CE 42.1 43.1 1.0
K C 177.0 177.5 0.5
K N 121.0 121.7 0.7
K Np na na 0.1
R H  7.81 7.81 0.0
R HA 3.26 3.19 -0.07
R HB 1.60 1.55 0.05
R HG 1.60 1.55 0.05
R HD 3.19 3.00 -0.19
R CA 58.4 58.6 0.2
R CB 34.4 35.2 0.9
R CG 27.2 28.1 1.0
R CD 43.8 44.3 0.5
R CZ 159.6 163.5 4.0
R C 185.8 186.1 0.2
R N 122.4 122.8 0.4
R NE 85.6 91.5 5.9
R NG 71.2 93.2 22
"""


def _parse_ph_shift_table():
    """Return {residue: {atom: delta}}; neighbours are keyed residue+'p'/'s'."""
    shifts = {}
    for line in PH_SHIFT_TABLE.strip().split("\n"):
        fields = line.split()
        residue, atom = fields[0], fields[1]
        shifts.setdefault(residue, {})[atom] = float(fields[4])
        if len(fields) > 6:
            shifts.setdefault(residue + "p", {})[atom] = float(fields[5])
            shifts.setdefault(residue + "s", {})[atom] = float(fields[6])
    return shifts


PH_SHIFTS = _parse_ph_shift_table()


def protonated_fraction(ph, pka, n_hill):
    """Henderson-Hasselbalch fraction of the protonated form."""
    return (10 ** (n_hill * (pka - ph))) / (1.0 + 10 ** (n_hill * (pka - ph)))


def _screened_coulomb(r, ionic_strength):
    kappa = np.sqrt(ionic_strength) / 3.08
    x = kappa * r / np.sqrt(6)
    return 332.286 * np.sqrt(6 / np.pi) * (
        1 - np.sqrt(np.pi) * x * np.exp(x ** 2) * erfc(x)
    ) / (DIELECTRIC * r)


def _energy_to_pka_units(energy, temperature):
    return energy * 4181.2 / (GAS_CONSTANT * temperature * np.log(10))


def _small_matrix_limits(ires, n_sites):
    ileft = max(1, ires - CUTOFF)
    iright = min(ileft + 2 * CUTOFF, n_sites)
    if iright == n_sites:
        ileft = max(1, iright - 2 * CUTOFF)
    return ileft, iright


def _small_matrix_position(ires, n_sites):
    resi = CUTOFF + 1
    if ires < CUTOFF + 1:
        resi = ires
    if ires > n_sites - CUTOFF:
        resi = min(n_sites, 2 * CUTOFF + 1) - (n_sites - ires)
    return resi


@lru_cache(maxsize=None)
def calc_pkas_from_seq(seq, temperature, ionic_strength):
    """
    Sequence-dependent pKa and Hill coefficient of every titratable site.

    seq must include the terminal markers, i.e. "n" + sequence + "c".
    Returns {position: (pKa, nH, residue)}, positions 0-based in the
    sequence without markers (the termini get -1 and len(sequence)).
    """
    phs = np.arange(1.99, 10.01, 0.15)

    pos = np.array([i for i, res in enumerate(seq) if res in PK0])
    n_sites = pos.shape[0]
    identity = np.diag(np.ones(n_sites))
    sites = "".join(seq[i] for i in pos)
    neg = np.array([i for i, res in enumerate(sites) if res in "DEYc"])
    seq_distance = np.array([abs(pos - pos[i]) for i in range(n_sites)])
    distance = DISTANCE_A + np.sqrt(seq_distance) * DISTANCE_B

    interaction = _screened_coulomb(distance, ionic_strength)
    interaction[identity == 1] = 0
    ww = _energy_to_pka_units(interaction, temperature) / 2

    charges_empty = np.zeros(n_sites)
    if len(neg):
        charges_empty[neg] = -1

    pk0s = np.array([PK0[res] for res in sites])
    nh0s = np.full(n_sites, 0.9)
    pkas, nhs = pk0s.copy(), nh0s.copy()

    titration = np.zeros((n_sites, len(phs)))
    small_n = min(2 * CUTOFF + 1, n_sites)
    all_states = np.array(
        [[int(c) for c in np.binary_repr(i, small_n)] for i in range(2 ** small_n)]
    )

    # Iterative fitting: every cycle uses the titration curves of the
    # previous one as the mean-field charges of the distant sites.
    for icycle in range(N_CYCLES):
        if icycle == 0:
            fraction_hold = np.array([
                protonated_fraction(ph, pk0s, nh0s) for ph in phs
            ])
        else:
            fraction_hold = titration.transpose()

        for ires in range(1, n_sites + 1):
            ileft, iright = _small_matrix_limits(ires, n_sites)
            resi = _small_matrix_position(ires, n_sites)
            selected = all_states[:, resi - 1] == 1

            for p, ph in enumerate(phs):
                fraction = fraction_hold[p].copy()
                fraction[ileft - 1: iright] = 0
                charges = charges_empty + fraction
                ww0 = np.diag(np.dot(ww, charges) * 2)
                g_full = ww + ww0 + ph * identity - np.diag(pk0s)
                g_small = g_full[ileft - 1: iright, ileft - 1: iright]

                weights = 10 ** -np.einsum(
                    "ti,ij,tj->t", all_states, g_small, all_states
                )
                titration[ires - 1, p] = weights[selected].sum() / weights.sum()

        for p in range(n_sites):
            try:
                (pkas[p], nhs[p]), _ = curve_fit(
                    protonated_fraction, phs, titration[p], [pk0s[p], nh0s[p]]
                )
            except RuntimeError:
                print(
                    f"[pH correction] pKa fit did not converge for "
                    f"{sites[p]} at position {pos[p] - 1}; keeping previous value.",
                    flush=True,
                )

    return {
        int(i) - 1: (pkas[p], nhs[p], seq[i])
        for p, i in enumerate(pos)
    }


def pka_key(seq, temperature, ionic_strength):
    """Key of PKA_CACHE; seq is the sequence without terminal markers."""
    return ("n" + seq + "c", float(temperature), max(0.0001, float(ionic_strength)))


def _fit_pkas(key):
    return key, calc_pkas_from_seq(*key)


def load_pkas(keys):
    """
    Make sure the pKa values of all keys are in PKA_CACHE: read the disk
    cache, fit the missing ones in parallel and save them.
    """
    if not PKA_CACHE and PKA_CACHE_FILE.exists():
        PKA_CACHE.update(pd.read_pickle(PKA_CACHE_FILE))

    missing = sorted(set(keys) - PKA_CACHE.keys())
    if not missing:
        return

    print(f"\n[pH correction] Fitting pKa values for {len(missing)} sequence/condition pairs.", flush=True)
    with Pool(min(len(missing), os.cpu_count())) as pool:
        PKA_CACHE.update(pool.map(_fit_pkas, missing))

    # Keep what other runs saved meanwhile; write to a temporary file first,
    # so parallel runs never read a partial cache.
    if PKA_CACHE_FILE.exists():
        for key, pkas in pd.read_pickle(PKA_CACHE_FILE).items():
            PKA_CACHE.setdefault(key, pkas)
    PKA_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = PKA_CACHE_FILE.with_suffix(f".{os.getpid()}.{datetime.now().timestamp()}.tmp")
    pd.to_pickle(dict(PKA_CACHE), tmp_file)
    tmp_file.replace(PKA_CACHE_FILE)


def get_pkas(seq, temperature, ionic_strength):
    """pKa values from PKA_CACHE, fitted here if missing (seq without markers)."""
    key = pka_key(seq, temperature, ionic_strength)
    if key not in PKA_CACHE:
        PKA_CACHE[key] = calc_pkas_from_seq(*key)
    return PKA_CACHE[key]


@lru_cache(maxsize=None)
def get_ph_offsets(seq, temperature, ph, ionic_strength, reference_ph=7.0,
                   intrinsic_reference=True, reference_temperature=None,
                   reference_ionic_strength=None):
    """
    Predicted shift change (ppm) at the sample pH relative to reference_ph.

    With intrinsic_reference the reference state uses the intrinsic pKa
    (as in POTENCI); otherwise it uses the sequence-dependent pKa and Hill
    coefficient at reference_temperature and reference_ionic_strength
    (default: those of the sample, so the offset is zero when
    ph == reference_ph).

    Returns a DataFrame indexed by 0-based sequence position with one
    column per classifier feature (HN, N, CO, CA, CB, HA, HB). The result
    is cached, so callers must not modify it.
    """
    pkas = get_pkas(seq, temperature, ionic_strength)
    reference_pkas = None
    if not intrinsic_reference:
        reference_pkas = get_pkas(
            seq,
            temperature if reference_temperature is None else reference_temperature,
            ionic_strength if reference_ionic_strength is None else reference_ionic_strength,
        )
    features = list(ATOM_TO_FEATURE.values())
    offsets = np.zeros((len(seq), len(features)))

    for i, (pka, n_hill, residue) in pkas.items():
        if residue in "nc":
            continue

        # The standard reference uses the intrinsic pKa, as in POTENCI.
        if intrinsic_reference:
            reference_pka, reference_n_hill = PK0[residue], n_hill
        else:
            reference_pka, reference_n_hill, _ = reference_pkas[i]
        jump = protonated_fraction(ph, pka, n_hill) - protonated_fraction(
            reference_ph, reference_pka, reference_n_hill
        )

        for j, atom in enumerate(ATOM_TO_FEATURE):
            if atom not in PH_SHIFTS[residue]:
                continue
            offsets[i, j] -= jump * PH_SHIFTS[residue][atom]

            for neighbour, suffix in ((i - 1, "p"), (i + 1, "s")):
                neighbour_shifts = PH_SHIFTS.get(residue + suffix, {})
                if atom in neighbour_shifts and 0 <= neighbour < len(seq):
                    offsets[neighbour, j] -= jump * neighbour_shifts[atom]

    return pd.DataFrame(offsets, columns=features)


def _get_sample_conditions(ent, shift_frame):
    """Return [(type, value, unit)] of the sample conditions of a shift list."""
    shift_list_ids = shift_frame.get_tag(
        "_Assigned_chem_shift_list.Sample_condition_list_ID"
    )

    if not shift_list_ids:
        return []

    condition_id = _clean_nmrstar_value(shift_list_ids[0])

    if condition_id is None:
        return []

    conditions = []
    for loop in ent.get_loops_by_category("Sample_condition_variable"):
        for cid, variable_type, value, unit in zip(
            loop.get_tag("_Sample_condition_variable.Sample_condition_list_ID"),
            loop.get_tag("_Sample_condition_variable.Type"),
            loop.get_tag("_Sample_condition_variable.Val"),
            loop.get_tag("_Sample_condition_variable.Val_units"),
        ):
            if _clean_nmrstar_value(cid) != condition_id:
                continue
            variable_type = _clean_nmrstar_value(variable_type)
            if variable_type is None:
                continue
            conditions.append((
                variable_type.lower(),
                _clean_nmrstar_value(value),
                _clean_nmrstar_value(unit),
            ))
    return conditions


def get_ph_for_shift_list(ent, shift_frame):
    """Sample pH of a chemical-shift saveframe, or None."""
    for variable_type, value, _ in _get_sample_conditions(ent, shift_frame):
        if variable_type not in {"ph", "ph*"} or value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def get_ionic_strength_for_shift_list(ent, shift_frame):
    """Ionic strength in M of a chemical-shift saveframe, or None."""
    for variable_type, value, unit in _get_sample_conditions(ent, shift_frame):
        if variable_type != "ionic strength" or value is None or unit is None:
            continue
        try:
            ionic_strength = float(value)
        except ValueError:
            continue

        unit_normalized = unit.strip().lower()
        if unit_normalized == "m":
            return ionic_strength
        if unit_normalized == "mm":
            return ionic_strength / 1000.0
    return None


def get_sequence_for_shift_list(ent, shift_frame):
    """
    One-letter sequence of the entity a shift list refers to, as
    {Seq_ID: residue}, taken from the Entity_poly_seq loop. Non-standard
    residues are "X". Returns None if the sequence cannot be found.
    """
    entity_ids = [
        _clean_nmrstar_value(eid)
        for eid in shift_frame["_Atom_chem_shift.Entity_ID"]
    ]
    entity_ids = [eid for eid in entity_ids if eid is not None]

    if not entity_ids:
        return None

    entity_id = entity_ids[0]

    sequence = {}
    for loop in ent.get_loops_by_category("Entity_poly_seq"):
        for eid, num, mon_id in zip(
            loop.get_tag("_Entity_poly_seq.Entity_ID"),
            loop.get_tag("_Entity_poly_seq.Num"),
            loop.get_tag("_Entity_poly_seq.Mon_ID"),
        ):
            if _clean_nmrstar_value(eid) != entity_id:
                continue
            try:
                sequence[int(num)] = AMINO_ACID_3_TO_1.get(
                    str(mon_id).strip().upper(), "X"
                )
            except ValueError:
                continue

    return sequence or None


def correct_ph_shifts(df, sequences, reference_ph=7.0, intrinsic_reference=True,
                      reference_temperature=None, reference_ionic_strength=None,
                      skip_termini=False, skip_neutral=False):
    """
    Normalize chemical shifts to a reference pH (see get_ph_offsets for
    intrinsic_reference, reference_temperature and reference_ionic_strength).

    df needs the columns 'protein', 'seq_id', 'amino', 'pH',
    'ionic_strength' and 'temperature_K'; sequences maps protein ->
    {Seq_ID: one-letter residue}. Proteins without a known pH or
    sequence, and residues whose type does not match the sequence,
    remain unchanged. With skip_termini the first and last residue of
    the sequence also remain unchanged, as in POTENCI. With skip_neutral
    proteins whose pH is within 0.01 of reference_ph remain unchanged
    (POTENCI corrects only if pH < 6.99 or pH > 7.01).
    """
    corrected = df.copy()

    for column in ["protein", "seq_id", "amino", "pH", "ionic_strength", "temperature_K"]:
        if column not in corrected.columns:
            raise ValueError(
                f"DataFrame must contain a '{column}' column."
            )

    features = [
        feature for feature in ATOM_TO_FEATURE.values()
        if feature in corrected.columns
    ]

    samples = []
    for protein, group in corrected.groupby("protein", sort=False):
        ph = pd.to_numeric(group["pH"], errors="coerce").iloc[0]
        sequence = sequences.get(protein)

        if pd.isna(ph) or not sequence:
            continue

        if skip_neutral and abs(float(ph) - float(reference_ph)) <= 0.01 + 1e-9:
            continue

        temperature = pd.to_numeric(group["temperature_K"], errors="coerce").iloc[0]
        if pd.isna(temperature):
            temperature = DEFAULT_TEMPERATURE

        ionic_strength = pd.to_numeric(group["ionic_strength"], errors="coerce").iloc[0]
        if pd.isna(ionic_strength):
            ionic_strength = DEFAULT_IONIC_STRENGTH

        # Seq_IDs need not start at 1 or be contiguous.
        seq_ids = sorted(sequence)
        position = {seq_id: i for i, seq_id in enumerate(seq_ids)}
        seq = "".join(sequence[seq_id] for seq_id in seq_ids)
        samples.append((group, float(ph), float(temperature), float(ionic_strength), seq, position))

    # Fit all pKa values needed below at once (in parallel, cached on disk).
    keys = [pka_key(seq, temperature, ionic_strength) for _, _, temperature, ionic_strength, seq, _ in samples]
    if not intrinsic_reference:
        keys += [
            pka_key(
                seq,
                temperature if reference_temperature is None else reference_temperature,
                ionic_strength if reference_ionic_strength is None else reference_ionic_strength,
            )
            for _, _, temperature, ionic_strength, seq, _ in samples
        ]
    load_pkas(keys)

    for group, ph, temperature, ionic_strength, seq, position in samples:
        offsets = get_ph_offsets(
            seq, temperature, ph, ionic_strength,
            reference_ph=float(reference_ph),
            intrinsic_reference=intrinsic_reference,
            reference_temperature=reference_temperature,
            reference_ionic_strength=reference_ionic_strength,
        )

        # Position of every row in the sequence (-1 if its Seq_ID is unknown),
        # kept only where the residue type agrees with the sequence.
        pos = (
            pd.to_numeric(group["seq_id"], errors="coerce")
            .map(position)
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )
        amino = (
            group["amino"]
            .astype("string")
            .str.strip()
            .str.upper()
            .map(AMINO_ACID_3_TO_1)
            .fillna("")
            .to_numpy(dtype=str)
        )
        valid = (pos >= 0) & (amino == np.array(list(seq))[pos])
        if skip_termini:
            valid &= (pos > 0) & (pos < len(seq) - 1)

        rows = group.index[valid]
        for feature in features:
            corrected.loc[rows, feature] = (
                pd.to_numeric(corrected.loc[rows, feature], errors="coerce")
                - offsets[feature].to_numpy()[pos[valid]]
            )

    return corrected
