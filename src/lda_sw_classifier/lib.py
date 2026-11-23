import hashlib
import inspect
import os
import sys
import warnings

from collections import Counter
from datetime import datetime
from functools import wraps
from itertools import compress, combinations
from time import time, sleep

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import pulp
import pynmrstar
import seaborn as sns
import toml
import torch

from pycirclize import Circos
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import resample
from pathlib import Path

warnings.filterwarnings("ignore")
from masserstein import NMRSpectrum, estimate_proportions # pylint: disable=wrong-import-position
from wnet import WassersteinNetwork, Distribution # pylint: disable=wrong-import-position
from wnet.distribution import Distribution_1D # pylint: disable=wrong-import-position
from wnet.distances import DistanceMetric # pylint: disable=wrong-import-position
from lda_sw_classifier.temperature import (
    get_temperature_for_shift_list,
    correct_temperature_shifts,
)
from lda_sw_classifier.ph import (
    get_ph_for_shift_list,
    get_ionic_strength_for_shift_list,
    get_sequence_for_shift_list,
    correct_ph_shifts,
    DEFAULT_IONIC_STRENGTH,
    DEFAULT_TEMPERATURE,
)


ALL_POSSIBLE = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]

NUM_COLORS = 20

# Downloaded BMRB entries are kept here, so LOO runs fetch every entry once.
BMRB_CACHE_DIR = Path(__file__).resolve().parents[2] / "Input" / "bmrb_cache"

# Assembled (and corrected) BMRB tables, reused by every LOO fold of a run.
BMRB_TABLE_CACHE = {}

CM = plt.get_cmap("Spectral")

############################## DECORATORS #####################################
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"Took: {(te-ts):2.4f} sec")
        return result
    return wrap

def task(f):
    @wraps(f)
    def wrap(*args, **kw):
        i = args[1]
        warnings.filterwarnings("ignore")
        print(f"{datetime.now().time()} Starting task {i}.", flush=True)
        s = time()
        res = f(*args, **kw)
        e = time()
        print(
            f"{datetime.now().time()} Finished task {i}. Total time = {(e-s):.4f} seconds.",
            flush=True
        )
        return res
    return wrap

############################## HELPER FUNC ####################################
def load_config(config_path):
    with open(config_path, "r") as file:
        config = toml.load(file)
    return config

def prepare_parameters(config):
    parameters = {
        "algo" : config["parameters"]["algo"],
        "n_projections": config["parameters"]["n_projections"],
        "seed": config["parameters"]["seed"],
        "k": config["parameters"]["k"],
        "device": config["parameters"]["device"],
        "with_bins": config["parameters"]["with_bins"],
        "sample_method": config["parameters"]["sample_method"],
        "vote_type": config["parameters"]["vote_type"],
        "scale": config["parameters"]["scale"],
        "eps": config["parameters"]["eps"],
        "kappa1": config["parameters"]["kappa1"],
        "kappa2": config["parameters"]["kappa2"],
        "note": config["parameters"]["note"],
        "nuclei": config["parameters"]["nuclei"],
        "temperature_correction": config["parameters"]["temperature_correction"],
        "reference_temperature": config["parameters"]["reference_temperature"],
        "classifier": config["parameters"].get("classifier", "lda"),
        "ph_correction": config["parameters"].get("ph_correction", False),
        "reference_ph": config["parameters"].get("reference_ph", 7.0),
        "correction_reference": config["parameters"].get("correction_reference", "standard"),
        "transport": config["parameters"].get("transport", "sw"),
        "bins_backend": config["parameters"].get("bins_backend", "magnetstein"),
    }
    # Any other value would silently skip the correction.
    if parameters["correction_reference"] not in ["standard", "test"]:
        raise ValueError(
            f"correction_reference must be 'standard' or 'test', "
            f"not '{parameters['correction_reference']}'"
        )
    # Only added when set, so output filenames of runs without it don't change.
    if "ph_skip_termini" in config["parameters"]:
        parameters["ph_skip_termini"] = config["parameters"]["ph_skip_termini"]
    if "ph_skip_neutral" in config["parameters"]:
        parameters["ph_skip_neutral"] = config["parameters"]["ph_skip_neutral"]
    # W cost metric (L1 | L2 | LINF), ET filter (all | certain) and combination rule
    # (tree | mix | assign) with its parameters; also only added when set.
    for key in [
        "w_metric", "et_filter", "combine", "combine_weight", "combine_floor", "combine_slack"
    ]:
        if key in config["parameters"]:
            parameters[key] = config["parameters"][key]
    print(parameters)
    return parameters

def lda_note(parameters):
    """Note used in LDA-only output filenames; ET+LDA runs get an `_et` suffix."""
    if parameters["classifier"] == "et_lda":
        return f"{parameters['note']}_et"
    return parameters["note"]

def fetch_bmrb_entry(bmrb_id):
    """Load a BMRB entry from the local cache, downloading it on first use."""
    cache_file = BMRB_CACHE_DIR / f"bmr{bmrb_id}.str"
    if cache_file.exists():
        return pynmrstar.Entry.from_file(str(cache_file))

    while True:
        try:
            ent = pynmrstar.Entry.from_database(bmrb_id)
            break
        except (ConnectionError, OSError):
            print(f"\nConnection error on: {bmrb_id}", flush=True)
            sleep(2)

    # Write to a temporary file first, so parallel runs never read a partial entry.
    BMRB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # The PID keeps the name unique when parallel runs fetch the same entry at once.
    tmp_file = cache_file.with_suffix(f".{os.getpid()}.{datetime.now().timestamp()}.tmp")
    ent.write_to_file(str(tmp_file))
    tmp_file.replace(cache_file)
    return ent

def bmrb_table_key(parameters, scanlist):
    """Hash of everything the assembled BMRB table depends on, including the code."""
    module_dir = Path(__file__).resolve().parent
    parts = [
        "\n".join(scanlist),
        str(parameters["temperature_correction"]),
        str(parameters["reference_temperature"]),
        str(parameters["ph_correction"]),
        str(parameters["reference_ph"]),
        str(parameters["correction_reference"]),
        str(parameters.get("ph_skip_termini", False)),
        str(parameters.get("ph_skip_neutral", False)),
        inspect.getsource(assemble_bmrb_table),
        (module_dir / "temperature.py").read_text(),
        (module_dir / "ph.py").read_text(),
    ]
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()[:16]

def read_bmrb_id(bmrbID_path):
    with open(bmrbID_path, "r") as f:
        scanlist = [line.rstrip("\n") for line in f]
    while scanlist[-1] == "":
        scanlist.pop(-1)
    return scanlist
############################## SW CLASSIFIER ###################################
# def normalize_matrix_col(matrix):
#     """
#     Function normalizing columns.
#     """
#     col_sums = np.sum(matrix, axis=0)
#     col_sums[col_sums == 0] = 1
#     matrix = matrix/col_sums
#     return matrix

# def normalize_matrix_rows(matrix):
#     """
#     Function normalising rows.
#     """
#     row_sums = np.sum(matrix, axis=1)
#     row_sums[row_sums == 0] = 1
#     matrix = matrix/row_sums[:, np.newaxis]
#     return matrix

def normalize_matrix_col(matrix, eps=1e-12):
    """
    Normalize columns and clip values to (eps, 1 - eps).
    """
    col_sums = np.sum(matrix, axis=0)
    col_sums[col_sums == 0] = 1
    matrix = matrix / col_sums

    # Clip to avoid exact 0 or 1
    matrix = np.clip(matrix, eps, 1 - eps)
    return matrix


def normalize_matrix_rows(matrix, eps=1e-12):
    """
    Normalize rows and clip values to (eps, 1 - eps).
    """
    row_sums = np.sum(matrix, axis=1)
    row_sums[row_sums == 0] = 1
    matrix = matrix / row_sums[:, np.newaxis]

    # Clip to avoid exact 0 or 1
    matrix = np.clip(matrix, eps, 1 - eps)
    return matrix

def get_random_projections(
    d, n_projections, seed=None, backend=None, type_as=None
):
    """
    Function returning n random d-dimensional vectors as a matrix (d x n).
    """
    if backend is None:
        nx = ot.backend.NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy": # pylint: disable=no-member
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))
    return projections

def get_projections(
    d, n_projections, dimPRO, dimGLY, n_specPRO,
    n_specGLY, seed=None, backend=None, type_as=None
):
    """
    Function returning n+2 d-dimensional vectors,
    2 first are suggested for PRO and GlY, the rest n random.
    """
    if backend is None:
        nx = ot.backend.NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy": # pylint: disable=no-member
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))

    device = projections.device
    if dimPRO is not None:
        vector_dimPRO = np.eye(d)[:, dimPRO].reshape(-1, 1)
        projections_dimPRO = nx.from_numpy(np.tile(vector_dimPRO, (1,  n_specPRO))).to(device)

    if dimGLY is not None:
        vector_dimGLY = np.eye(d)[:, dimGLY].reshape(-1, 1)
        projections_dimGLY = nx.from_numpy(np.tile(vector_dimGLY, (1,  n_specGLY))).to(device)

    if dimGLY is not None and dimPRO is not None:
        # pylint: disable=possibly-used-before-assignment
        combined_projections = nx.concatenate(
            [projections_dimPRO, projections_dimGLY, projections], axis=1
        )
    elif dimGLY is not None:
        combined_projections = nx.concatenate([projections_dimGLY, projections], axis=1)
    elif dimPRO is not None:
        combined_projections = nx.concatenate([projections_dimPRO, projections], axis=1)
    else:
        combined_projections = projections

    return combined_projections

def bins_transport(projected1, projected2, kappa1=2.1, kappa2=2.1, balance=None):
    """
    Function returning transport plan using transport from magnetstein
    """
    # Chemical shifts with uniform intensities
    if balance:
        confs1 = list(zip(projected1.detach().numpy(), balance))
    else:
        confs1 = [(shift, 1) for shift in projected1.detach().numpy()]
    confs2 = [(shift, 1) for shift in projected2.detach().numpy()]

    sp1_confs = confs1.copy()
    sp2_confs = confs2.copy()

    # Creating and normalising spectra
    spectrum1 = NMRSpectrum(confs=confs1)
    spectrum2 = NMRSpectrum(confs=confs2)
    spectrum1.normalize()
    spectrum2.normalize()

    # Estimation
    estimation = estimate_proportions(
        spectrum1, [spectrum2], what_to_compare="area", MTD=kappa1, MTD_th=kappa2,
        verbose=False, solver=pulp.LpSolverDefault
    )

    common_horizontal_axis = estimation["common_horizontal_axis"]

    updated = []
    if len(spectrum1.confs) != len(estimation["common_horizontal_axis"]) or \
                len(spectrum2.confs) != len(estimation["common_horizontal_axis"]):
        for sp in [spectrum1, spectrum2]:
            missing = set(
                estimation["common_horizontal_axis"]
            ).difference([el[0] for el in sp.confs])
            new_confs = sp.confs + [(missing_point, 0.) for missing_point in missing]
            updated.append(NMRSpectrum(confs=new_confs))
        spectrum1, spectrum2 = updated # pylint: disable=unbalanced-tuple-unpacking

    noise_in_spectrum1 = estimation["noise"]
    noise_in_spectrum2 = estimation["noise_in_components"]

    p0 = 1-sum(estimation["proportions"])
    p0_prime = estimation["proportion_of_noise_in_components"]
    spectrum1.normalize(target_value=1-p0_prime)
    spectrum2.normalize(target_value=1-p0)
    if np.array([el[1] for el in spectrum1.confs]).shape != np.array(noise_in_spectrum1).shape:
        return np.zeros((len(confs1), len(confs2)))
    intensities1 = list(np.array([el[1] for el in spectrum1.confs]) - np.array(noise_in_spectrum1))
    spectrum1_no_noise = NMRSpectrum(confs = list(zip(common_horizontal_axis, intensities1)))
    if np.array([el[1] for el in spectrum2.confs]).shape != np.array(noise_in_spectrum2).shape:
        return np.zeros((len(confs1), len(confs2)))
    intensities2 = list(np.array([el[1] for el in spectrum2.confs]) - np.array(noise_in_spectrum2))
    spectrum2_no_noise = NMRSpectrum(confs = list(zip(common_horizontal_axis, intensities2)))
    transport_plan_bins = list(spectrum1_no_noise.WSDistanceMoves(spectrum2_no_noise))


    dict1_non = {sp1_confs[i][0]: i for i in range(len(sp1_confs))}
    dict2_non =  {sp2_confs[j][0]: j for j in range(len(sp2_confs))}

    # Transport plan
    transport_plan = np.zeros((len(confs1), len(confs2)))
    for i, j, v in transport_plan_bins:
        if dict1_non.get(j, -1) != -1 and dict2_non.get(i, -1) != -1:
            if 1-p0-p0_prime != 0:
                transport_plan[dict1_non[j]][dict2_non[i]] = v/(1-p0-p0_prime)
            else:
                continue
    return transport_plan

def wnet_bins_transport(projected1, projected2, kappa1=2.1, kappa2=2.1):
    """
    Function returning transport plan using wnet (the magnetstein model: auxiliary points
    with costs kappa1 and kappa2, proportion of the second spectrum optimised)
    """
    x1 = projected1.detach().numpy().astype(np.float64)
    x2 = projected2.detach().numpy().astype(np.float64)
    network = WassersteinNetwork(
        Distribution_1D(x1, np.full(len(x1), 1 / len(x1))),
        [Distribution_1D(x2, np.full(len(x2), 1 / len(x2)))],
        DistanceMetric.L1, force_dense_1d=True, cost_precision_bits=30
    )
    network.add_independent_asymmetric_trash(kappa1, kappa2)
    network.build()

    # The cost is convex in the proportion, so bisect on the sign of its derivative
    low, high = 0.0, 2.0
    network.solve([high])
    while network.spectrum_proportion_derivatives()[0] <= 0:
        low, high = high, 2 * high
        network.solve([high])
    for _ in range(30):
        mid = (low + high) / 2
        network.solve([mid])
        if network.spectrum_proportion_derivatives()[0] > 0:
            high = mid
        else:
            low = mid
    network.solve([(low + high) / 2])

    idx1, idx2, flows = network.flows_for_target(0)
    transport_plan = np.zeros((len(x1), len(x2)))
    np.add.at(transport_plan, (np.asarray(idx1), np.asarray(idx2)), np.asarray(flows))
    # The optimal 1D plan is not unique; take the monotone one (as magnetstein and ot.emd_1d)
    transport_plan = monotone_plan(x1, transport_plan.sum(axis=1), x2, transport_plan.sum(axis=0))
    # Normalised by the transported mass, as in bins_transport
    if transport_plan.sum() > 0:
        transport_plan = transport_plan / transport_plan.sum()
    return transport_plan

def monotone_plan(x1, mass1, x2, mass2):
    """
    Function returning the monotone (north-west corner) transport plan between 1D marginals
    """
    order1, order2 = np.argsort(x1, kind="stable"), np.argsort(x2, kind="stable")
    left1, left2 = mass1[order1].astype(np.float64), mass2[order2].astype(np.float64)
    transport_plan = np.zeros((len(x1), len(x2)))
    i = j = 0
    while i < len(order1) and j < len(order2):
        moved = min(left1[i], left2[j])
        transport_plan[order1[i], order2[j]] += moved
        left1[i] -= moved
        left2[j] -= moved
        if left1[i] <= 1e-15 * max(1.0, mass1.sum()):
            i += 1
        if left2[j] <= 1e-15 * max(1.0, mass2.sum()):
            j += 1
    return transport_plan

def do_sw(
    X_s, X_t, n_projections, seed, device
) :
    """
    Function making projections on these selected vectors.
    """
    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            f"X_s and X_t must have the same number of dimensions {X_s.shape[1]} \
              and {X_t.shape[1]} respectively given"
        )
    nx = ot.backend.get_backend(X_s, X_t)

    d = X_s.shape[1]

    projections = get_random_projections(d, n_projections, seed, backend=nx, type_as=X_s).to(device)
    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)

    return {
        "projections": projections,
        "X_s_projections": X_s_projections,
        "X_t_projections": X_t_projections
    }

def do_sw_spec(
    X_s, X_t, n_projections, dimPRO, dimGLY,
    n_specPRO, n_specGLY, seed, device
) :
    """
    Function making projections on these selected vectors (with special cases for PRO and GLY).
    """
    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            f"X_s and X_t must have the same number of dimensions {X_s.shape[1]} and \
             {X_t.shape[1]} respectively given"
        )
    nx = ot.backend.get_backend(X_s, X_t)

    d = X_s.shape[1]

    projections = get_projections(d, n_projections, dimPRO, dimGLY, n_specPRO, n_specGLY,\
                                                 seed, backend=nx, type_as=X_s).to(device)
    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)
    return {
        "projections": projections,
        "X_s_projections": X_s_projections,
        "X_t_projections": X_t_projections
    }

def voting(
    n_projections, s_labels, classes, X_s_projections, X_t_projections, X_s_dim0,
    X_t_dim0, scale, kappa1=2.1, kappa2=2.1, k = None, with_bins = False,
    bins_backend = "magnetstein"
):
    """
    Function returning list of labels and total transport plan for classes
    (voting by total transport).
    """
    # Count array – how many times for each pair (p_s, p_t) of points from source to target
    # it was the transport with the highest mass
    counts = np.zeros((X_s_dim0, X_t_dim0))

   # Array holding the cumulative transport plan (sum of EMDs across all projections)
    total_transport = np.zeros((X_s_dim0, X_t_dim0))

    # Array holding the cumulative transport plan for the target class
    # (EMD summed over all projections)
    total_transport_classes = np.zeros((len(classes), X_t_dim0))

    # Iterating over all projections
    for i in range(n_projections):
        if with_bins and X_s_projections[:,i].nelement() != 0 and \
                X_t_projections[:,i].nelement() != 0:
            transport_fn = wnet_bins_transport if bins_backend == "wnet" else bins_transport
            emd = transport_fn(
                X_s_projections[:,i], X_t_projections[:,i], kappa1=kappa1, kappa2=kappa2
            )
            total_transport = total_transport + emd
        else:
            emd = ot.emd_1d(X_s_projections[:,i], X_t_projections[:,i])
            total_transport = total_transport + ot.backend.to_numpy(emd)

        # Identifying indices of source points that contributed
        # the most mass to the given target point
        max_values = np.argmax(emd, axis=0)
        for j in range(counts.shape[1]):
            counts[max_values[j],j] += 1

    test_labels = []
    train_count = {i:s_labels.tolist().count(i) for i in s_labels.tolist()}

    # If we're considering all
    if not k:
        for i in range(X_t_dim0):
            # Dictionary "d" will store, for a given target point,
            # the total transport from each class
            d = {}
            for j in range(X_s_dim0):
                d[s_labels[j]] = d.get(s_labels[j], 0) + total_transport[j, i]
                total_transport_classes[classes.index(s_labels[j]),i] += total_transport[j, i]
            if scale:
                d = {i:v/train_count[i] for i, v in d.items()}

           # Assign the class with the highest total transport
            max_key = max(d, key=d.get)
            if d[max_key] > 0:
                test_labels.append(max_key)
            else:
                test_labels.append("None")
    else:
        top_k_values_indices = []
        if scale:
            division_factors = [
                train_count[s_labels[coln]] for coln in range(total_transport.shape[1])
                ]
            total_transport = total_transport / division_factors
        for col in range(total_transport.shape[1]):
            indices = np.argpartition(total_transport[:, col], -k)[-k:]
            top_k_values_indices.append(indices)

        # Consider only the top-k values of total transport
        for i in range(len(top_k_values_indices)):
            d = {}
            for j in range(k):
                ind = top_k_values_indices[i][j]
                d[s_labels[ind]] = d.get(s_labels[ind], 0) + total_transport[ind, i]
                total_transport_classes[classes.index(s_labels[ind]),i] += total_transport[ind, i]
            max_key = max(d, key=d.get)
            if  d[max_key] > 0:
                test_labels.append(max_key)
            else:
                test_labels.append("None")

    return test_labels, total_transport_classes

# Denumeration of the index
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return None

def voting_spec(
    n_projections, n_specPRO, n_specGLY, s_labels, t_counts, classes, X_s_projections,
    X_t_projections, X_s_dim0, X_t_dim0, parameters_amino, scale, kappa1=2.1,
    kappa2=2.1, k = None, with_bins = False, bins_backend = "magnetstein"
):
    # Expected number of residues of each class still to assign (from the sequence,
    # content["count_rest"]), not the true labels of the test set
    # t_labels = list(t_labels)
    # lab_dict = {lab:t_labels.count(lab) for lab in set(t_labels)}
    lab_dict = {lab: cnt for lab, cnt in t_counts.items() if cnt > 0}
    t_labels = list(lab_dict)
    train_lab_dict = {
        lab : [id for id, l in enumerate(s_labels) if l == lab] for lab in set(s_labels)
    }

    indexes_s = set(range(X_s_dim0))
    indexes_t = set(range(X_t_dim0))
    enumerated_s = dict(enumerate(sorted(list(indexes_s))))
    enumerated_t = dict(enumerate(sorted(list(indexes_t))))

    # Array holding the cumulative transport plan for target-source
    # (sum of EMDs across all projections)
    total_transport = np.zeros((X_s_dim0, X_t_dim0))

    # Array holding the cumulative transport plan for target-class
    # (sum of EMDs across all projections)
    total_transport_classes = np.zeros((len(classes), X_t_dim0))

    forsure_dict = {lab:[] for lab in set(t_labels)}
    proportions = {itx: {} for itx in range(X_t_dim0)}
    labels_forsure = {itx: None for itx in range(X_t_dim0)}

    train_count = {i:s_labels.tolist().count(i) for i in s_labels.tolist()}

    for i in range(n_projections + n_specPRO + n_specGLY):
        # Check PRO and remove certain points / if empty, remove the class
        if i == n_specPRO and "PRO" in t_labels and "PRO" in s_labels:
            indices_to_drop_t = []
            enum_indices_to_drop_t = []
            best_indexes = []
            if not k:
                for ik in range(len(enumerated_t.items())):
                    # In the dictionary "d", we will store the total transport
                    # from a given class for a target point
                    d = {}
                    for j in range(len(enumerated_s.items())):
                        if s_labels[enumerated_s[j]] in t_labels and \
                                lab_dict[s_labels[enumerated_s[j]]]:
                            d[s_labels[enumerated_s[j]]] = \
                                d.get(s_labels[enumerated_s[j]], 0) + total_transport[j, ik]
                            cidx = classes.index(s_labels[enumerated_s[j]])
                            total_transport_classes[cidx, enumerated_t[ik]] += \
                                total_transport[j, ik]
                    if scale:
                        d = {id:v/train_count[id] for id, v in d.items()}
                    # Assign the class of the point from which the transport was the greatest
                    if not d:
                        break
                    max_key = max(d, key=d.get)
                    sum_key = sum(d.values())
                    if d[max_key] > 0 and max_key == "PRO":
                        best_indexes.append(((d[max_key]/sum_key, max_key), ik))
                best_indexes.sort(reverse=True)
                bound = min(len(best_indexes), lab_dict["PRO"])
                indexes = [
                    v for k,v in best_indexes[:bound] if k and k[0] >= parameters_amino[k[1]]
                ]
                lab_dict["PRO"] = lab_dict.get("PRO") - len(indexes)
                for index in indexes:
                    forsure_dict.get("PRO",[]).append(enumerated_t[index])
                    proportions[enumerated_t[index]] = d
                    labels_forsure[enumerated_t[index]] = "PRO"
                    indices_to_drop_t.append(index)
                    enum_indices_to_drop_t.append(enumerated_t[index])
            else:
                total_elements = total_transport.shape[0]
                if k <= 0 or k > total_elements:
                    ki = min(total_elements, max(1, k))
                else:
                    ki = k
                top_k_values_indices = []
                if scale:
                    division_factors = [
                        train_count[s_labels[coln]] for coln in range(total_transport.shape[1])
                    ]
                    total_transport = total_transport / division_factors
                for col in range(total_transport.shape[1]):
                    indices = np.argpartition(total_transport[:, col], -ki)[-ki:]
                    top_k_values_indices.append(indices)

               # Consider only the top k highest values of total transport
                for ik in range(len(top_k_values_indices)):
                    d = {}
                    for j in range(ki):
                        ind = top_k_values_indices[ik][j]
                        if s_labels[enumerated_s[ind]] in t_labels and \
                                lab_dict[s_labels[enumerated_s[ind]]]>0:
                            d[s_labels[enumerated_s[ind]]] = \
                                d.get(s_labels[enumerated_s[ind]], 0) + total_transport[ind, ik]
                            cidx = classes.index(s_labels[enumerated_s[ind]])
                            total_transport_classes[cidx, enumerated_t[ik]] += \
                                total_transport[ind, ik]
                    if not d:
                        break
                    max_key = max(d, key=d.get)
                    sum_key = sum(d.values())
                    if d[max_key] > 0 and max_key == "PRO":
                        best_indexes.append(((d[max_key]/sum_key, max_key), ik))
                best_indexes.sort(reverse=True)
                bound = min(len(best_indexes), lab_dict["PRO"])
                indexes = [
                    v for k,v in best_indexes[:bound] if k and k[0] >= parameters_amino[k[1]]
                ]
                lab_dict["PRO"] = lab_dict.get("PRO") - len(indexes)
                for index in indexes:
                    forsure_dict.get("PRO",[]).append(enumerated_t[index])
                    proportions[enumerated_t[index]] = d
                    labels_forsure[enumerated_t[index]] = "PRO"
                    indices_to_drop_t.append(index)
                    enum_indices_to_drop_t.append(enumerated_t[index])
            X_t_projections = np.delete(X_t_projections, indices_to_drop_t, axis=0)
            total_transport = np.delete(total_transport, indices_to_drop_t, axis=1)
            indexes_t = indexes_t.difference(set(enum_indices_to_drop_t))
            # Removing from the training set
            if lab_dict.get("PRO", 0) == 0:
                indices_to_drop_s = [
                    k for k,v in enumerated_s.items() if v in train_lab_dict["PRO"]
                ]
                X_s_projections = np.delete(X_s_projections, indices_to_drop_s, axis=0)
                total_transport = np.delete(total_transport, indices_to_drop_s, axis=0)
                indexes_s = indexes_s.difference(train_lab_dict["PRO"])
                enumerated_s = dict(enumerate(sorted(list(indexes_s))))
            enumerated_t = dict(enumerate(sorted(list(indexes_t))))

        # Check GLY and remove certain points / if empty, remove the class
        if i == n_specPRO + n_specGLY and "GLY" in t_labels and "GLY" in s_labels:
            indices_to_drop_t = []
            enum_indices_to_drop_t = []
            best_indexes = []
            if not k:
                for ik in range(len(enumerated_t.items())):
                    # In the dictionary "d", we will store the total transport
                    # from the given class for a target point
                    d = {}
                    for j in range(len(enumerated_s.items())):
                        if s_labels[enumerated_s[j]] in t_labels and \
                                lab_dict[s_labels[enumerated_s[j]]]:
                            d[s_labels[enumerated_s[j]]] = \
                                d.get(s_labels[enumerated_s[j]], 0) + total_transport[j, ik]
                            cidx = classes.index(s_labels[enumerated_s[j]])
                            total_transport_classes[cidx, enumerated_t[ik]] += \
                                total_transport[j, ik]
                    if scale:
                        d = {id:v/train_count[id] for id, v in d.items()}
                    # Assign the class of the point from which the transport was the largest
                    if not d:
                        break
                    max_key = max(d, key=d.get)
                    sum_key = sum(d.values())
                    if d[max_key] > 0 and max_key == "GLY":
                        best_indexes.append(((d[max_key]/sum_key, max_key), ik))
                best_indexes.sort(reverse=True)
                bound = min(len(best_indexes), lab_dict["GLY"])
                indexes = [
                    v for k,v in best_indexes[:bound] if k and k[0] >= parameters_amino[k[1]]
                ]
                lab_dict["GLY"] = lab_dict.get("GLY") - len(indexes)
                for index in indexes:
                    forsure_dict.get("GLY",[]).append(enumerated_t[index])
                    proportions[enumerated_t[index]] = d
                    labels_forsure[enumerated_t[index]] = "GLY"
                    indices_to_drop_t.append(index)
                    enum_indices_to_drop_t.append(enumerated_t[index])
            else:
                total_elements = total_transport.shape[0]
                if k <= 0 or k > total_elements:
                    ki = min(total_elements, max(1, k))
                else:
                    ki = k
                top_k_values_indices = []
                if scale:
                    division_factors = [
                        train_count[s_labels[coln]] for coln in range(total_transport.shape[1])
                    ]
                    total_transport = total_transport / division_factors
                for col in range(total_transport.shape[1]):
                    indices = np.argpartition(total_transport[:, col], -ki)[-ki:]
                    top_k_values_indices.append(indices)

                # Consider only the top k highest values of total transport
                for ik in range(len(top_k_values_indices)):
                    d = {}
                    for j in range(ki):
                        ind = top_k_values_indices[ik][j]
                        if s_labels[enumerated_s[ind]] in t_labels and \
                                lab_dict[s_labels[enumerated_s[ind]]]>0:
                            d[s_labels[enumerated_s[ind]]] = \
                                d.get(s_labels[enumerated_s[ind]], 0) + total_transport[ind, ik]
                            cidx = classes.index(s_labels[enumerated_s[ind]])
                            total_transport_classes[cidx, enumerated_t[ik]] += \
                                total_transport[ind, ik]
                    if not d:
                        break
                    max_key = max(d, key=d.get)
                    sum_key = sum(d.values())
                    if d[max_key] > 0 and max_key == "GLY":
                        best_indexes.append(((d[max_key]/sum_key, max_key), ik))
                best_indexes.sort(reverse=True)
                bound = min(len(best_indexes), lab_dict["GLY"])

                indexes = [
                    v for k,v in best_indexes[:bound] if k and k[0] >= parameters_amino[k[1]]
                ]
                lab_dict["GLY"] = lab_dict.get("GLY") - len(indexes)
                for index in indexes:
                    forsure_dict.get("GLY",[]).append(enumerated_t[index])
                    proportions[enumerated_t[index]] = d
                    labels_forsure[enumerated_t[index]] = "GLY"
                    indices_to_drop_t.append(index)
                    enum_indices_to_drop_t.append(enumerated_t[index])
            X_t_projections = np.delete(X_t_projections, indices_to_drop_t, axis=0)
            total_transport = np.delete(total_transport, indices_to_drop_t, axis=1)
            indexes_t = indexes_t.difference(set(enum_indices_to_drop_t))
            # Removing from the training set
            if lab_dict.get("GLY", 0) == 0:
                indices_to_drop_s = [
                    k for k,v in enumerated_s.items() if v in train_lab_dict["GLY"]
                ]
                X_s_projections = np.delete(X_s_projections, indices_to_drop_s, axis=0)
                total_transport = np.delete(total_transport, indices_to_drop_s, axis=0)
                indexes_s = indexes_s.difference(train_lab_dict["GLY"])
                enumerated_s = dict(enumerate(sorted(list(indexes_s))))
            enumerated_t = dict(enumerate(sorted(list(indexes_t))))

        if with_bins and X_s_projections[:,i].nelement() != 0 and \
                X_t_projections[:,i].nelement() != 0:
            transport_fn = wnet_bins_transport if bins_backend == "wnet" else bins_transport
            emd = transport_fn(
                X_s_projections[:,i], X_t_projections[:,i], kappa1=kappa1, kappa2=kappa2
                )
            total_transport = total_transport + emd
        else:
            if not enumerated_t or not enumerated_s:
                break
            emd = ot.emd_1d(X_s_projections[:,i], X_t_projections[:,i])
            total_transport = total_transport + ot.backend.to_numpy(emd)

        # Check if we can proceed to the next step or remove the class from training
        # (when no certain points)
        if i - (n_specPRO + n_specGLY) > 1000:
            # If we consider all
            indices_to_drop_t = []
            indices_to_drop_s = []
            enum_indices_to_drop_t = []
            enum_indices_to_drop_s = []
            if not k:
                for ik in range(len(enumerated_t.items())):
                    # In the dictionary "d", we will store the total transport
                    # from the given class for a target point
                    d = {}
                    for j in range(len(enumerated_s.items())):
                        if s_labels[enumerated_s[j]] in t_labels and \
                                lab_dict[s_labels[enumerated_s[j]]] > 0:
                            d[s_labels[enumerated_s[j]]] = \
                                d.get(s_labels[enumerated_s[j]], 0) + total_transport[j, ik]
                            cidx = classes.index(s_labels[enumerated_s[j]])
                            total_transport_classes[cidx ,enumerated_t[ik]] += \
                                total_transport[j, ik]
                    if scale:
                        d = {id:v/train_count[id] for id, v in d.items()}

                    # Assign the class from which the total transport was the largest
                    if not d:
                        break
                    max_key = max(d, key=d.get)
                    sum_key = sum(d.values())

                    if d[max_key]/sum_key > parameters_amino[max_key] and lab_dict[max_key] > 0:
                        forsure_dict[max_key].append(enumerated_t[ik])
                        proportions[enumerated_t[ik]] = d
                        labels_forsure[enumerated_t[ik]] = max_key
                        lab_dict[max_key] = lab_dict.get(max_key) - 1
                        indices_to_drop_t.append(ik)
                        enum_indices_to_drop_t.append(enumerated_t[ik])
                    if lab_dict.get(max_key, 0) == 0:
                        indices_to_drop_s += [
                            k for k,v in enumerated_s.items() if v in train_lab_dict[max_key]
                        ]
                        enum_indices_to_drop_s += list(train_lab_dict[max_key])
            else:
                total_elements = total_transport.shape[0]
                if k <= 0 or k > total_elements:
                    ki = min(total_elements, max(1, k))
                else:
                    ki = k
                top_k_values_indices = []
                if scale:
                    division_factors = [
                        train_count[s_labels[enumerated_s[coln]]]
                        for coln in range(total_transport.shape[1])
                    ]
                    total_transport = total_transport / division_factors
                for col in range(total_transport.shape[1]):
                    indices = np.argpartition(total_transport[:, col], -ki)[-ki:]
                    top_k_values_indices.append(indices)

                # Consider only the top k highest values of total transport
                for ik in range(len(top_k_values_indices)):
                    d = {}
                    for j in range(ki):
                        ind = top_k_values_indices[ik][j]
                        if s_labels[enumerated_s[ind]] in t_labels and \
                                lab_dict[s_labels[enumerated_s[ind]]] > 0:
                            d[s_labels[enumerated_s[ind]]] = \
                                d.get(s_labels[enumerated_s[ind]], 0) + total_transport[ind, ik]
                            cidx = classes.index(s_labels[enumerated_s[ind]])
                            total_transport_classes[cidx, enumerated_t[ik]] += \
                                total_transport[ind, ik]
                    if not d:
                        break
                    max_key = max(d, key=d.get)
                    sum_key = sum(d.values())

                    if d[max_key]/sum_key > parameters_amino[max_key] and lab_dict[max_key] > 0:
                        forsure_dict[max_key].append(enumerated_t[ik])
                        proportions[enumerated_t[ik]] = d
                        labels_forsure[enumerated_t[ik]] = max_key
                        lab_dict[max_key] = lab_dict.get(max_key) - 1
                        indices_to_drop_t.append(ik)
                        enum_indices_to_drop_t.append(enumerated_t[ik])

                    if lab_dict.get(max_key, 0) == 0:
                        indices_to_drop_s += \
                            [k for k,v in enumerated_s.items() if v in train_lab_dict[max_key]]
                        enum_indices_to_drop_s += list(train_lab_dict[max_key])

            X_t_projections = np.delete(X_t_projections, indices_to_drop_t, axis=0)
            total_transport = np.delete(total_transport, indices_to_drop_t, axis=1)
            indexes_t = indexes_t.difference(set(enum_indices_to_drop_t))
            X_s_projections = np.delete(X_s_projections, indices_to_drop_s, axis=0)
            total_transport = np.delete(total_transport, indices_to_drop_s, axis=0)

            indexes_s = indexes_s.difference(set(enum_indices_to_drop_s))
            enumerated_s = dict(enumerate(sorted(list(indexes_s))))
            enumerated_t = dict(enumerate(sorted(list(indexes_t))))

        # Exit the loop and want to distribute what is left
        if not enumerated_t or not enumerated_s:
            break

    # If we consider all
    test_labels = []
    if not k:
        for i in range(X_t_dim0):
            if labels_forsure[i]:
                test_labels.append(labels_forsure[i])
            else:
                ix = get_key(enumerated_t, i)
                # In the dictionary "d", we will store the total transport
                # from the given class for a target point
                d = {}
                for j in range(len(enumerated_s.items())):
                    d[s_labels[enumerated_s[j]]] = \
                        d.get(s_labels[enumerated_s[j]], 0) + total_transport[j, ix]
                    cidx = classes.index(s_labels[enumerated_s[j]])
                    total_transport_classes[cidx ,enumerated_t[ix]] += total_transport[j, ix]
                if scale:
                    d = {id:v/train_count[id] for id, v in d.items()}

               # Assign the class from which the total transport was the largest
                if not d:
                    break

                max_key = max(d, key=d.get)
                if d[max_key] > 0:
                    test_labels.append(max_key)
                else:
                    test_labels.append("None")

    else:
        for i in range(X_t_dim0):
            if labels_forsure[i]:
                test_labels.append(labels_forsure[i])
            else:
                ix = get_key(enumerated_t, i)
                total_elements = total_transport.shape[0]
                if k <= 0 or k > total_elements:
                    ki = min(total_elements, max(1, k))
                else:
                    ki = k
                top_k_values_indices = []
                if scale:
                    division_factors = [
                        train_count[s_labels[coln]] for coln in range(total_transport.shape[1])
                    ]
                    total_transport = total_transport / division_factors
                for col in range(total_transport.shape[1]):
                    indices = np.argpartition(total_transport[:, col], -ki)[-ki:]
                    top_k_values_indices.append(indices)
                d = {}
                for j in range(ki):
                    ind = top_k_values_indices[ix][j]
                    d[s_labels[enumerated_s[ind]]] = \
                        d.get(s_labels[enumerated_s[ind]], 0) + total_transport[ind, ix]
                    cidx = classes.index(s_labels[enumerated_s[ind]]),enumerated_t[ix]
                    total_transport_classes[cidx] += total_transport[ind, ix]
                if not d:
                    break

                max_key = max(d, key=d.get)
                if  d[max_key] > 0:
                    test_labels.append(max_key)
                else:
                    test_labels.append("None")
    return test_labels, total_transport_classes

def classify_sw(
    X_s, X_t, s_labels, classes, n_projections, seed, device, scale, kappa1=2.1, kappa2=2.1,
    k = None, with_bins = False, bins_backend = "magnetstein", printing=False
):
    proj_dict = do_sw(X_s, X_t, n_projections, seed, device)
    X_s_projections = proj_dict["X_s_projections"].cpu().detach()
    X_t_projections = proj_dict["X_t_projections"].cpu().detach()
    directions = proj_dict["projections"].cpu().detach()
    if printing:
        print(f"X_s_projections[:,0]: {X_s_projections[:,0]}\n \
                X_t_projections[:,0]: {X_t_projections[:,0]}\n directions[0]: {directions[0]}\n")
        print(f"X_s_projections[:,1]: {X_s_projections[:,1]}\n \
                X_t_projections[:,1]: {X_t_projections[:,1]}\n directions[1]: {directions[1]}\n")

    return voting(
        n_projections, s_labels, classes, X_s_projections, X_t_projections, X_s.shape[0],
        X_t.shape[0], scale, kappa1=kappa1, kappa2=kappa2, k = k, with_bins = with_bins,
        bins_backend = bins_backend
    )

def classify_sw_spec(
    X_s, X_t, s_labels, t_counts, classes, n_projections, dimPRO, dimGLY, n_specPRO, n_specGLY,
    parameters_amino, seed, device, scale, kappa1=2.1, kappa2=2.1, k = None, with_bins = False,
    bins_backend = "magnetstein", printing=False
):
    proj_dict = do_sw_spec(
        X_s, X_t, n_projections, dimPRO, dimGLY, n_specPRO, n_specGLY, seed, device
    )
    X_s_projections = proj_dict["X_s_projections"].cpu().detach()
    X_t_projections = proj_dict["X_t_projections"].cpu().detach()
    directions = proj_dict["projections"].cpu().detach()
    if printing:
        print(f"X_s_projections[:,0]: {X_s_projections[:,0]}\n \
                X_t_projections[:,0]: {X_t_projections[:,0]}\n directions[0]: {directions[0]}\n")
        print(f"X_s_projections[:,1]: {X_s_projections[:,1]}\n \
                X_t_projections[:,1]: {X_t_projections[:,1]}\n directions[1]: {directions[1]}\n")

    return voting_spec(
        n_projections, n_specPRO, n_specGLY, s_labels, t_counts,
        classes, X_s_projections, X_t_projections, X_s.shape[0], X_t.shape[0], parameters_amino,
        scale, kappa1=kappa1, kappa2=kappa2, k = k, with_bins = with_bins,
        bins_backend = bins_backend
    )


def accuracy(check, labels):
    len_all = check.shape[0]
    li = 0
    yes_no = []
    for i in range(len_all):
        if check[i] == labels[i]:
            li += 1
            yes_no.append("green")
        elif labels[i] == "None":
            len_all -= 1
            yes_no.append("blue")
        else:
            yes_no.append("red")
    if len_all == 0:
        len_all = 1
    return li / len_all, yes_no

def count_neighbours(group, nuclei, eps):
    # --- Single nucleus case ---
    if len(nuclei) == 1:
        col = nuclei[0]
        group_subset = group[[col]]
        distances = cdist(group_subset.values, group_subset.values, metric="euclidean")
        neighbours = (distances <= eps).sum(axis=1)
        group["neighbours"] = neighbours
        return group

    # return group
    all_comb = []
    for l in range(1, len(nuclei)):
        all_comb += combinations(nuclei, l)
    for comb in all_comb:
        try:
            group_subset = group[[*comb]]
            distances = np.zeros((group_subset.shape[0], group_subset.shape[0]))
            distances += cdist(group_subset.values, group_subset.values, metric="euclidean")
            neighbours = (distances <= eps).sum(axis=1)
        except KeyError:
            continue
    group["neighbours"] = neighbours / len(all_comb)
    return group

def find_best_reference(AATs_count, train_t, nuclei, eps):
    sampled_df = pd.DataFrame()
    for amino, group in train_t.groupby("amino"):
        group_with_counts = count_neighbours(
            group, nuclei, eps).sort_values(["neighbours"], ascending=False
        )
        sampled_group = group_with_counts.head(AATs_count[amino])
        sampled_df = pd.concat([sampled_df, sampled_group])
    return sampled_df
############################## W CLASSIFIER ####################################
def classify_w(
    X_s, X_t, s_labels, classes, kappa1=2.1, kappa2=2.1, with_bins=False, metric="L2",
    class_mass=None
):
    """
    Function returning list of labels and total transport plan for classes
    (multidimensional Wasserstein distance computed with wnet, no projections).
    With class_mass ({class: share}) every class gets that share of the train mass,
    split evenly among its points; otherwise all train points get the same mass.
    """
    total_transport_classes = np.zeros((len(classes), X_t.shape[0]))
    if X_s.shape[0] == 0:
        return ["None"] * X_t.shape[0], total_transport_classes

    if class_mass is None:
        # Uniform intensities normalised to 1, as in SW
        source_mass = np.full(X_s.shape[0], 1 / X_s.shape[0])
    else:
        labels = pd.Series(s_labels)
        source_mass = (
            labels.map(class_mass).fillna(0.0) / labels.map(labels.value_counts())
        ).to_numpy(dtype=np.float64)
        source_mass = source_mass / source_mass.sum()
    source = Distribution(X_s.T, source_mass)
    target = Distribution(X_t.T, np.full(X_t.shape[0], 1 / X_t.shape[0]))
    # Chemical shift differences are below 1 ppm, so costs must not be rounded to integers
    network = WassersteinNetwork(
        source, [target], getattr(DistanceMetric, metric), cost_precision_bits=30
    )
    if with_bins:
        # Auxiliary points as in magnetstein: unmatched train mass costs kappa1, test mass kappa2
        network.add_independent_asymmetric_trash(kappa1, kappa2)
    else:
        # Costlier than any transport, so it only absorbs intensity rounding
        diameter = np.linalg.norm(
            np.ptp(np.vstack([X_s, X_t]), axis=0), ord=1 if metric == "L1" else None
        )
        network.add_simple_trash(2 * diameter + 1)
    network.build()
    network.solve()

    s_idx, t_idx, flows = network.flows_for_target(0)
    transport_plan = np.zeros((X_s.shape[0], X_t.shape[0]))
    np.add.at(transport_plan, (np.asarray(s_idx), np.asarray(t_idx)), np.asarray(flows))

    for j, lab in enumerate(s_labels):
        total_transport_classes[classes.index(lab)] += transport_plan[j]

    # Test points transported entirely to auxiliary points stay unclassified
    test_labels = [
        classes[np.argmax(col)] if col.max() > 0 else "None" for col in total_transport_classes.T
    ]
    return test_labels, total_transport_classes

################################### LDA ########################################

def use_extra_trees(parameters, nuclei_meas):
    """ET is used when at least two of CA, HB, CB are measured (classifier = et_lda)."""
    n_best = sum(nucleus in nuclei_meas for nucleus in ["CA", "HB", "CB"])
    return parameters["classifier"] == "et_lda" and n_best >= 2

def fixed_et_rows(parameters, used_et, probs):
    """
    ET rows kept out of SW/W: all of them (et_filter = "all"), or with
    et_filter = "certain" only those with top probability 1, as for LDA rows.
    """
    if parameters.get("et_filter", "all") == "certain":
        return used_et & (probs.max(axis=1) == 1)
    return used_et

def get_classifier(parameters, nuclei_meas):
    """ExtraTrees or LDA, depending on the measured nuclei."""
    if use_extra_trees(parameters, nuclei_meas):
        return ExtraTreesClassifier(
            n_estimators=1000,
            class_weight="balanced_subsample",
            max_features=0.5,
            n_jobs=-1,
            random_state=parameters["seed"],
        )
    return LDA()


# def lda_classification_validate(parameters, protein, scanlist, output_dir):
#     content = {}
#     algo = parameters["algo"]
#     note = parameters["note"]
#     reference_temperature = parameters["reference_temperature"]
#     temperature_correction = parameters["temperature_correction"]

#     names = [
#         "H", "HB", "HB1", "HB2", "HB3",
#         "CA", "CB", "C", "CO", "N",
#         "amino", "protein", "temperature_K",
#     ]
#     TableB = pd.DataFrame(columns=names)

#     for k, l in enumerate(scanlist):
#         waiter = True

#         sys.stdout.write(
#             f"\nScanning data from BMRB: {k + 1} of {len(scanlist)}"
#         )
#         sys.stdout.flush()

#         while waiter:
#             try:
#                 ent = pynmrstar.Entry.from_database(l)
#                 waiter = False
#             except (ConnectionError, OSError):
#                 print(f"\nConnection error on: {l}", flush=True)
#                 sleep(2)

#         spectral_peaks = ent.get_saveframes_by_category(
#             "assigned_chemical_shifts"
#         )

#         if not spectral_peaks:
#             continue

#         shift_frame = spectral_peaks[0]

#         temperature_K = get_temperature_for_shift_list(
#             ent,
#             shift_frame,
#         )

#         TableC = pd.DataFrame(columns=names)

#         w = shift_frame["_Atom_chem_shift.Seq_ID"]
#         x = shift_frame["_Atom_chem_shift.Comp_ID"]
#         y = shift_frame["_Atom_chem_shift.Atom_ID"]
#         z = shift_frame["_Atom_chem_shift.Val"]

#         if not (len(w) == len(x) == len(y) == len(z)):
#             raise ValueError(
#                 f"Inconsistent chemical-shift arrays for BMRB {l}: "
#                 f"Seq_ID={len(w)}, Comp_ID={len(x)}, "
#                 f"Atom_ID={len(y)}, Val={len(z)}"
#             )

#         if not w:
#             continue

#         wp = w[0]
#         n = 0

#         for i, seq_id in enumerate(w):
#             if seq_id != wp:
#                 n += 1
#                 wp = seq_id

#             TableC.at[n, "protein"] = l
#             TableC.at[n, "amino"] = x[i]
#             TableC.at[n, "temperature_K"] = temperature_K

#             atom_name = str(y[i]).strip()
#             shift_value = pd.to_numeric(z[i], errors="coerce")

#             if pd.isna(shift_value):
#                 continue

#             if atom_name in names:
#                 TableC.at[n, atom_name] = float(shift_value)

#         TableB = pd.concat(
#             [TableB, TableC],
#             ignore_index=True,
#         )

#     print("\nFinished assembling BMRB data.", flush=True)

#     # Assemble residue-level features.
#     TableA = pd.DataFrame(index=TableB.index)

#     TableA["amino"] = TableB["amino"]
#     TableA["protein"] = TableB["protein"]

#     # Preserve experimental temperature for every residue.
#     TableA["temperature_K"] = pd.to_numeric(
#         TableB["temperature_K"],
#         errors="coerce",
#     )

#     # Average available beta-proton shifts.
#     hb_columns = ["HB", "HB1", "HB2", "HB3"]
#     TableA["HB"] = TableB[hb_columns].apply(
#         pd.to_numeric, errors="coerce"
#     ).mean(axis=1)

#     # Map atom names to classifier feature names.
#     TableA["HN"] = pd.to_numeric(
#         TableB["H"], errors="coerce"
#     )
#     TableA["CA"] = pd.to_numeric(
#         TableB["CA"], errors="coerce"
#     )
#     TableA["CB"] = pd.to_numeric(
#         TableB["CB"], errors="coerce"
#     )
#     TableA["N"] = pd.to_numeric(
#         TableB["N"], errors="coerce"
#     )

#     # The coefficient table must use the exact key "CO".
#     # In BMRB, the carbonyl carbon atom is usually named "C".
#     TableA["CO"] = pd.to_numeric(
#         TableB["C"], errors="coerce"
#     )

#     # Average available alpha-proton shifts.
#     ha_columns = ["HA", "HA2", "HA3"]
#     available_ha_columns = [
#         col for col in ha_columns if col in TableB.columns
#     ]

#     if available_ha_columns:
#         TableA["HA"] = TableB[available_ha_columns].apply(
#             pd.to_numeric, errors="coerce"
#         ).mean(axis=1)
#     else:
#         TableA["HA"] = np.nan

#     # Correct chemical shifts before scaling and classification.
#     if temperature_correction:
#         TableA = correct_temperature_shifts(
#             TableA,
#             reference_temperature=reference_temperature,
#         )

#     AAT_dict = {
#         "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
#         "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
#         "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
#         "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
#         "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
#     }

#     nuclei = parameters["nuclei"]

#     # Only selected nuclei enter the classifier.
#     # Metadata columns remain outside the numerical feature matrix.
#     train_data = TableA[
#         nuclei + ["amino", "protein"]
#     ].copy()

#     protein_filter = {}
#     considered = train_data[
#         train_data["protein"].isin(scanlist)
#     ]

#     check_labels = considered.loc[
#         considered["protein"] == protein, "amino"
#     ].values

#     AATs_unique = sorted(set(check_labels))
#     AATs_count = Counter(check_labels)

#     AATs_missing = list(
#         set(AAT_dict.values()) - set(AATs_unique)
#     )

#     data_considered = considered.copy()

#     for AAT in AATs_missing:
#         data_considered = data_considered.drop(
#             data_considered.query("amino == @AAT").index
#         )

#     train_t = data_considered.loc[
#         data_considered["protein"] != protein
#     ].copy()

#     test_t = data_considered.loc[
#         data_considered["protein"] == protein
#     ].copy()

#     # Scale numerical features. Keep labels and protein IDs unchanged.
#     for_scaling = pd.concat(
#         [test_t, train_t], ignore_index=True
#     )

#     for col in nuclei:
#         values = pd.to_numeric(
#             for_scaling[col], errors="coerce"
#         )
#         mean = values.mean()
#         std = values.std(ddof=0)

#         if pd.isna(std) or std == 0:
#             for_scaling[col] = values - mean
#         else:
#             for_scaling[col] = (
#                 values - mean
#             ) / np.sqrt(std)

#     n_test = len(test_t)
#     test_t = for_scaling.iloc[:n_test].copy()
#     train_t = for_scaling.iloc[n_test:].copy()

#     # Bootstrap balancing for applicable algorithms.
#     train_data = train_t.copy()
#     balanced_data = []

#     for amino_acid in train_data["amino"].unique():
#         class_data = train_data[
#             train_data["amino"] == amino_acid
#         ]
#         n_samples = train_data["amino"].value_counts().max()

#         balanced_class_data = resample(
#             class_data,
#             replace=True,
#             n_samples=n_samples,
#             random_state=parameters["seed"],
#         )
#         balanced_data.append(balanced_class_data)

#     if balanced_data:
#         balanced_train_data = pd.concat(
#             balanced_data, ignore_index=True
#         )
#     else:
#         balanced_train_data = train_t.copy()

#     test_t = test_t.dropna(
#         subset=nuclei, how="all"
#     )
#     train_t = train_t.dropna(
#         subset=nuclei, how="all"
#     )

#     if algo in ["lda_bootstrap", "filtered_bootstrap"]:
#         train_t = balanced_train_data.dropna(
#             subset=nuclei, how="all"
#         )

#     train_labels = train_t.loc[
#         train_t["protein"] != protein, "amino"
#     ].values

#     check_labels = test_t.loc[
#         test_t["protein"] == protein, "amino"
#     ].values

#     test_lab = [None for _ in range(test_t.shape[0])]
#     probs = np.ndarray(
#         shape=(len(test_lab), len(AATs_unique))
#     )

#     test_nan = np.array(test_t.isnull())
#     combs_nan = np.unique(test_nan, axis=0)

#     if algo == "sw":
#         content["train"] = train_t
#         content["test"] = test_t
#         content["check_labels"] = check_labels
#         content["train_labels"] = train_labels
#         content["count_rest"] = AATs_count
#         return content

#     # Get column positions of GLY and PRO.
#     if "GLY" in AATs_unique:
#         G_idx = AATs_unique.index("GLY")
#     if "PRO" in AATs_unique:
#         P_idx = AATs_unique.index("PRO")

#     # Loop over combinations of missing nuclei.
#     for i in combs_nan:
#         idxs_nan = np.all(test_nan == i, axis=1)
#         nuclei_meas = list(compress(nuclei, ~i))
#         test_missing_comb = test_t.loc[
#             idxs_nan, nuclei_meas
#         ]

#         if np.all(i):
#             a = np.empty(
#                 (idxs_nan.sum(), probs.shape[1])
#             )
#             a[:] = np.nan
#             probs[idxs_nan, :] = a
#             continue

#         # Residues that could be GLY or PRO.
#         if (
#             {"HB", "CB", "HN"}.isdisjoint(nuclei_meas)
#             and all(x in AATs_unique for x in ["GLY", "PRO"])
#         ):
#             lda = LDA()
#             idxs = train_t[nuclei_meas].isnull().any(axis=1)
#             train_set_aux = train_t.loc[
#                 ~idxs, nuclei_meas
#             ].copy()
#             labels = train_labels[~idxs].copy()

#             lda.fit(train_set_aux, labels)
#             lda.means_ = 1
#             probs[idxs_nan, :] = lda.predict_proba(
#                 test_missing_comb
#             )
#             test_labels = list(
#                 lda.predict(test_missing_comb)
#             )

#             for idx, val in enumerate(idxs_nan):
#                 if val:
#                     test_lab[idx] = test_labels.pop(0)

#         # Residues that could be GLY.
#         elif (
#             {"HB", "CB"}.isdisjoint(set(nuclei_meas))
#             and "GLY" in AATs_unique
#         ):
#             lda = LDA()
#             train_set_aux = train_t.loc[
#                 train_t["amino"] != "PRO", nuclei_meas
#             ].copy()
#             labels = train_labels[
#                 train_labels != "PRO"
#             ].copy()

#             idxs = train_set_aux.isnull().any(axis=1)
#             train_set_aux = train_set_aux.loc[~idxs]
#             labels = labels[~idxs]

#             lda.fit(train_set_aux, labels)
#             probs_aux = lda.predict_proba(test_missing_comb)
#             test_labels = list(lda.predict(test_missing_comb))

#             if "PRO" in AATs_unique:
#                 probs[idxs_nan, :] = np.c_[
#                     probs_aux[:, :P_idx],
#                     np.zeros((idxs_nan.sum(), 1)),
#                     probs_aux[:, P_idx:],
#                 ]
#             else:
#                 probs[idxs_nan, :] = probs_aux

#             for idx, val in enumerate(idxs_nan):
#                 if val:
#                     test_lab[idx] = test_labels.pop(0)

#         # Residues that could be PRO.
#         elif (
#             {"HN"}.isdisjoint(set(nuclei_meas))
#             and "PRO" in AATs_unique
#         ):
#             lda = LDA()
#             train_set_aux = train_t.loc[
#                 train_t["amino"] != "GLY", nuclei_meas
#             ].copy()
#             labels = train_labels[
#                 train_labels != "GLY"
#             ].copy()

#             idxs = train_set_aux.isnull().any(axis=1)
#             train_set_aux = train_set_aux.loc[~idxs]
#             labels = labels[~idxs]

#             lda.fit(train_set_aux, labels)
#             probs_aux = lda.predict_proba(test_missing_comb)
#             test_labels = list(lda.predict(test_missing_comb))

#             if "PRO" not in set(AATs_unique).difference(set(labels)):
#                 if "GLY" in AATs_unique:
#                     probs[idxs_nan, :] = np.c_[
#                         probs_aux[:, :G_idx],
#                         np.zeros((idxs_nan.sum(), 1)),
#                         probs_aux[:, G_idx:],
#                     ]
#                 else:
#                     probs[idxs_nan, :] = probs_aux
#             else:
#                 if "GLY" in AATs_unique:
#                     probs[idxs_nan, :] = np.c_[
#                         probs_aux[:, :G_idx],
#                         np.zeros((idxs_nan.sum(), 1)),
#                         probs_aux[:, G_idx:P_idx - 1],
#                         np.zeros((idxs_nan.sum(), 1)),
#                         probs_aux[:, P_idx - 1:],
#                     ]
#                 else:
#                     probs[idxs_nan, :] = np.c_[
#                         probs_aux[:, :P_idx],
#                         np.zeros((idxs_nan.sum(), 1)),
#                         probs_aux[:, P_idx:],
#                     ]

#             for idx, val in enumerate(idxs_nan):
#                 if val:
#                     test_lab[idx] = test_labels.pop(0)

#         # Other residues.
#         else:
#             lda = LDA()
#             train_set_aux = train_t.query(
#                 'amino != "PRO" and amino != "GLY"'
#             )[nuclei_meas].copy()

#             labels = train_labels[
#                 np.logical_and(
#                     train_labels != "PRO",
#                     train_labels != "GLY",
#                 )
#             ].copy()

#             idxs = train_set_aux.isnull().any(axis=1)
#             train_set_aux = train_set_aux.loc[~idxs]
#             labels = labels[~idxs]

#             lda.fit(train_set_aux, labels)
#             probs_aux = lda.predict_proba(test_missing_comb)
#             test_labels = list(lda.predict(test_missing_comb))

#             if all(x in AATs_unique for x in ["GLY", "PRO"]):
#             #     probs[idxs_nan, :] = np.c_[
#             #         probs_aux[:, :G_idx],
#             #         np.zeros((idxs_nan.sum(), 1)),
#             #         probs_aux[:, G_idx:P_idx - 1],
#             #         np.zeros((idxs_nan.sum(), 1)),
#             #         probs_aux[:, P_idx - 1:],
#             #     ]
#             # elif "PRO" in AATs_unique:
#             #     probs[idxs_nan, :] = np.c_[
#             #         probs_aux[:, :P_idx],
#             #         np.zeros((idxs_nan.sum(), 1)),
#             #         probs_aux[:, P_idx:],
#             #     ]
#             # elif "GLY" in AATs_unique:
#             #     probs[idxs_nan, :] = np.c_[
#             #         probs_aux[:, :G_idx],
#             #         np.zeros((idxs_nan.sum(), 1)),
#             #         probs_aux[:, G_idx:],
#             #     ]

#             # Align auxiliary LDA probabilities to the destination
#             # amino-acid columns. PRO and GLY were excluded from training,
#             # so their probabilities are zero.
#                 aux_classes = list(lda.classes_)
#                 target_classes = list(AATs_unique)

#                 probs_missing = np.zeros(
#                     (idxs_nan.sum(), len(target_classes)),
#                     dtype=probs.dtype,
#                 )

#                 for aux_idx, aa in enumerate(aux_classes):
#                     if aa in target_classes:
#                         target_idx = target_classes.index(aa)
#                         probs_missing[:, target_idx] = probs_aux[:, aux_idx]

#                 probs[idxs_nan, :] = probs_missing

#             else:
#                 probs[idxs_nan, :] = probs_aux

#             for idx, val in enumerate(idxs_nan):
#                 if val:
#                     test_lab[idx] = test_labels.pop(0)

#     df_probs = pd.DataFrame(
#         probs, index=test_t.index, columns=AATs_unique
#     )

#     df_protein = pd.DataFrame({
#         "real labels": check_labels,
#         "classified as": test_lab,
#         "LDA score": df_probs.max(axis=1).values,
#     })

#     content["train"] = train_t
#     content["test"] = test_t
#     content["check_labels"] = check_labels
#     content["train_labels"] = train_labels
#     content["lda_labels"] = df_protein
#     content["lda_probs"] = df_probs
#     content["lda_accuracy"] = accuracy(
#         check_labels, test_lab
#     )

#     if algo in ["lda", "lda_bootstrap"]:
#         df_protein.to_csv(
#             Path(output_dir)
#             / f"labels_{protein}_{algo}_{note}.csv"
#         )
#         df_probs.to_csv(
#             Path(output_dir)
#             / f"probabilities_{protein}_{algo}_{note}.csv"
#         )

#     if algo in ["filtered", "filtered_bootstrap"]:
#         fltr = [
#             i
#             for i, a in enumerate(AATs_unique)
#             if AATs_count.get(a, 0) / len(check_labels) > 0.01
#         ]
#         protein_filter[protein] = (
#             probs[:, fltr].max(axis=1) == 1
#         )
#         filter_p = pd.DataFrame(
#             protein_filter[protein],
#             columns=[f"{protein}"],
#         )
#         content["filter"] = filter_p
#         content["count_rest"] = AATs_count - Counter(
#             [
#                 lab
#                 for i, lab in enumerate(test_lab)
#                 if protein_filter[protein][i]
#             ]
#         )

#     return content


def assemble_bmrb_table(parameters, scanlist):
    """
    Residue-level shift table for scanlist and the sequences {protein: {Seq_ID: residue}}.
    With correction_reference = "standard" the shifts are corrected to the reference
    temperature/pH here; with "test" they are left uncorrected (see correct_to_test_conditions).
    """
    key = bmrb_table_key(parameters, scanlist)
    if key in BMRB_TABLE_CACHE:
        TableA, sequences = BMRB_TABLE_CACHE[key]
        return TableA.copy(), sequences

    cache_file = BMRB_CACHE_DIR / "tables" / f"{key}.pkl"
    if cache_file.exists():
        print(f"\nLoading assembled BMRB data from {cache_file}", flush=True)
        TableA, sequences = pd.read_pickle(cache_file)
        BMRB_TABLE_CACHE[key] = (TableA, sequences)
        return TableA.copy(), sequences

    reference_temperature = parameters["reference_temperature"]
    temperature_correction = parameters["temperature_correction"]
    reference_ph = parameters["reference_ph"]
    ph_correction = parameters["ph_correction"]
    sequences = {}

    names = [
        "H", "HA", "HA2", "HA3", "HB", "HB1", "HB2", "HB3",
        "CA", "CB", "C", "CO", "N",
        "amino", "protein", "temperature_K",
        "seq_id", "pH", "ionic_strength",
    ]
    TableB = pd.DataFrame(columns=names)

    # Download and assemble BMRB residue-level data.
    for k, l in enumerate(scanlist):
        sys.stdout.write(
            f"\nScanning data from BMRB: {k + 1} of {len(scanlist)}"
        )
        sys.stdout.flush()

        ent = fetch_bmrb_entry(l)

        spectral_peaks = ent.get_saveframes_by_category(
            "assigned_chemical_shifts"
        )

        if not spectral_peaks:
            continue

        shift_frame = spectral_peaks[0]

        temperature_K = get_temperature_for_shift_list(
            ent, shift_frame
        )
        pH = get_ph_for_shift_list(ent, shift_frame)
        ionic_strength = get_ionic_strength_for_shift_list(
            ent, shift_frame
        )
        sequences[l] = get_sequence_for_shift_list(ent, shift_frame)

        TableC = pd.DataFrame(columns=names)

        w = shift_frame["_Atom_chem_shift.Seq_ID"]
        x = shift_frame["_Atom_chem_shift.Comp_ID"]
        y = shift_frame["_Atom_chem_shift.Atom_ID"]
        z = shift_frame["_Atom_chem_shift.Val"]

        if not (len(w) == len(x) == len(y) == len(z)):
            raise ValueError(
                f"Inconsistent chemical-shift arrays for BMRB {l}: "
                f"Seq_ID={len(w)}, Comp_ID={len(x)}, "
                f"Atom_ID={len(y)}, Val={len(z)}"
            )

        if not w:
            continue

        wp = w[0]
        n = 0

        for i, seq_id in enumerate(w):
            if seq_id != wp:
                n += 1
                wp = seq_id

            TableC.at[n, "protein"] = l
            TableC.at[n, "amino"] = x[i]
            TableC.at[n, "temperature_K"] = temperature_K
            TableC.at[n, "seq_id"] = seq_id
            TableC.at[n, "pH"] = pH
            TableC.at[n, "ionic_strength"] = ionic_strength

            atom_name = str(y[i]).strip()
            shift_value = pd.to_numeric(z[i], errors="coerce")

            if pd.isna(shift_value):
                continue

            if atom_name in names:
                TableC.at[n, atom_name] = float(shift_value)

        TableB = pd.concat(
            [TableB, TableC],
            ignore_index=True,
        )

    print("\nFinished assembling BMRB data.", flush=True)

    # Assemble residue-level features.
    TableA = pd.DataFrame(index=TableB.index)

    TableA["amino"] = TableB["amino"]
    TableA["protein"] = TableB["protein"]

    TableA["temperature_K"] = pd.to_numeric(
        TableB["temperature_K"],
        errors="coerce",
    )
    TableA["seq_id"] = TableB["seq_id"]
    TableA["pH"] = pd.to_numeric(
        TableB["pH"],
        errors="coerce",
    )
    TableA["ionic_strength"] = pd.to_numeric(
        TableB["ionic_strength"],
        errors="coerce",
    )

    # Average available beta-proton shifts.
    hb_columns = ["HB", "HB1", "HB2", "HB3"]
    TableA["HB"] = TableB[hb_columns].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)

    # Map atom names to classifier feature names.
    TableA["HN"] = pd.to_numeric(
        TableB["H"], errors="coerce"
    )
    TableA["CA"] = pd.to_numeric(
        TableB["CA"], errors="coerce"
    )
    TableA["CB"] = pd.to_numeric(
        TableB["CB"], errors="coerce"
    )
    TableA["N"] = pd.to_numeric(
        TableB["N"], errors="coerce"
    )

    # BMRB usually names the carbonyl carbon "C".
    TableA["CO"] = pd.to_numeric(
        TableB["C"], errors="coerce"
    )

    # Average available alpha-proton shifts.
    ha_columns = ["HA", "HA2", "HA3"]
    available_ha_columns = [
        col for col in ha_columns if col in TableB.columns
    ]

    if available_ha_columns:
        TableA["HA"] = TableB[available_ha_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
    else:
        TableA["HA"] = np.nan

    standard_reference = parameters["correction_reference"] == "standard"

    # Apply temperature correction before scaling.
    if temperature_correction and standard_reference:
        TableA = correct_temperature_shifts(
            TableA,
            reference_temperature=reference_temperature,
        )

    if ph_correction and standard_reference:
        TableA = correct_ph_shifts(
            TableA,
            sequences,
            reference_ph=reference_ph,
            skip_termini=parameters.get("ph_skip_termini", False),
            skip_neutral=parameters.get("ph_skip_neutral", False),
        )

    # Write to a temporary file first, so parallel runs never read a partial table.
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(f".{os.getpid()}.{datetime.now().timestamp()}.tmp")
    pd.to_pickle((TableA, sequences), tmp_file)
    tmp_file.replace(cache_file)
    BMRB_TABLE_CACHE[key] = (TableA, sequences)
    return TableA.copy(), sequences

def correct_to_test_conditions(TableA, sequences, protein, parameters):
    """
    Correct every protein to the temperature and pH of the test protein, which itself
    stays unchanged. If the test protein has no temperature (pH), that correction is skipped.
    """
    test_rows = TableA["protein"] == protein

    if parameters["temperature_correction"]:
        test_temperature = pd.to_numeric(
            TableA.loc[test_rows, "temperature_K"], errors="coerce"
        ).iloc[0]
        if pd.notna(test_temperature):
            TableA = correct_temperature_shifts(
                TableA,
                reference_temperature=float(test_temperature),
            )

    if parameters["ph_correction"]:
        test_ph = pd.to_numeric(
            TableA.loc[test_rows, "pH"], errors="coerce"
        ).iloc[0]
        if pd.notna(test_ph):
            # The pKa of the reference state is fitted at the temperature and ionic strength
            # of the test protein.
            test_conditions = TableA.loc[test_rows, ["temperature_K", "ionic_strength"]].apply(
                pd.to_numeric, errors="coerce"
            ).iloc[0]
            reference_temperature = (
                DEFAULT_TEMPERATURE if pd.isna(test_conditions["temperature_K"])
                else float(test_conditions["temperature_K"])
            )
            reference_ionic_strength = (
                DEFAULT_IONIC_STRENGTH if pd.isna(test_conditions["ionic_strength"])
                else float(test_conditions["ionic_strength"])
            )
            TableA = correct_ph_shifts(
                TableA,
                sequences,
                reference_ph=float(test_ph),
                intrinsic_reference=False,
                reference_temperature=reference_temperature,
                reference_ionic_strength=reference_ionic_strength,
                skip_termini=parameters.get("ph_skip_termini", False),
                skip_neutral=parameters.get("ph_skip_neutral", False),
            )

    return TableA

def lda_classification_validate(parameters, protein, scanlist, output_dir):
    content = {}
    algo = parameters["algo"]
    note = lda_note(parameters)
    TableA, sequences = assemble_bmrb_table(parameters, scanlist)

    if parameters["correction_reference"] == "test":
        TableA = correct_to_test_conditions(TableA, sequences, protein, parameters)

    AAT_dict = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
        "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
        "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
        "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
        "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }

    nuclei = parameters["nuclei"]

    # Keep only selected nuclei and metadata.
    train_data = TableA[
        nuclei + ["amino", "protein"]
    ].copy()

    protein_filter = {}

    considered = train_data[
        train_data["protein"].isin(scanlist)
    ]

    check_labels = considered.loc[
        considered["protein"] == protein, "amino"
    ].values

    AATs_unique = sorted(set(check_labels))
    AATs_count = Counter(check_labels)

    # Exclude amino acids absent from the target protein.
    AATs_missing = list(
        set(AAT_dict.values()) - set(AATs_unique)
    )

    data_considered = considered.copy()

    for AAT in AATs_missing:
        data_considered = data_considered.drop(
            data_considered.query("amino == @AAT").index
        )

    train_t = data_considered.loc[
        data_considered["protein"] != protein
    ].copy()

    test_t = data_considered.loc[
        data_considered["protein"] == protein
    ].copy()

    # Scale numerical features while leaving labels and IDs unchanged.
    # Preserve the original scaling convention.
    for_scaling = pd.concat(
        [test_t, train_t],
        ignore_index=True,
    )

    for col in nuclei:
        values = pd.to_numeric(
            for_scaling[col], errors="coerce"
        )

        mean = values.mean()
        std = values.std(ddof=0)

        if pd.isna(std) or std == 0:
            for_scaling[col] = values - mean
        else:
            for_scaling[col] = (
                values - mean
            ) / np.sqrt(std)

    n_test = len(test_t)

    test_t = for_scaling.iloc[:n_test].copy()
    train_t = for_scaling.iloc[n_test:].copy()

    # Prepare bootstrap-balanced training data.
    balanced_data = []

    for amino_acid in train_t["amino"].unique():
        class_data = train_t[
            train_t["amino"] == amino_acid
        ]

        n_samples = train_t["amino"].value_counts().max()

        balanced_class_data = resample(
            class_data,
            replace=True,
            n_samples=n_samples,
            random_state=parameters["seed"],
        )

        balanced_data.append(balanced_class_data)

    if balanced_data:
        balanced_train_data = pd.concat(
            balanced_data,
            ignore_index=True,
        )
    else:
        balanced_train_data = train_t.copy()

    # Remove residues with no measured selected nuclei.
    test_t = test_t.dropna(
        subset=nuclei, how="all"
    )

    train_t = train_t.dropna(
        subset=nuclei, how="all"
    )

    if algo in ["lda_bootstrap", "filtered_bootstrap"]:
        train_t = balanced_train_data.dropna(
            subset=nuclei, how="all"
        )

    # Recompute labels after filtering so their order matches test_t.
    check_labels = test_t["amino"].to_numpy()

    train_labels = train_t["amino"].to_numpy()

    # Initialize the output and probability matrix.
    test_lab = [None] * len(test_t)

    probs = np.full(
        (len(test_t), len(AATs_unique)),
        np.nan,
        dtype=float,
    )

    # IMPORTANT: identify missing-feature patterns using only the
    # numerical feature columns, not amino-acid labels or protein IDs.
    test_nan = test_t[nuclei].isnull().to_numpy()
    combs_nan = np.unique(test_nan, axis=0)

    # Rows classified by ExtraTrees (classifier = et_lda); never passed to SW.
    used_et = np.zeros(len(test_t), dtype=bool)

    # Return training/testing data directly for sliced Wasserstein.
    if algo == "sw":
        content["train"] = train_t
        content["test"] = test_t
        content["check_labels"] = check_labels
        content["train_labels"] = train_labels
        content["count_rest"] = AATs_count
        return content

    # Map model probabilities into the destination class order.
    # Classes absent from the auxiliary model receive probability zero.
    def align_probabilities(probabilities, classes):
        aligned = np.zeros(
            (probabilities.shape[0], len(AATs_unique)),
            dtype=float,
        )

        target_indices = {
            aa: idx for idx, aa in enumerate(AATs_unique)
        }

        for source_idx, aa in enumerate(classes):
            if aa in target_indices:
                aligned[:, target_indices[aa]] = (
                    probabilities[:, source_idx]
                )

        return aligned

    # Loop over combinations of missing nuclei.
    for missing_pattern in combs_nan:
        idxs_nan = np.all(
            test_nan == missing_pattern,
            axis=1,
        )

        nuclei_meas = list(
            compress(nuclei, ~missing_pattern)
        )

        test_missing_comb = test_t.loc[
            idxs_nan, nuclei_meas
        ]

        # No measured nuclei: no classification is possible.
        if np.all(missing_pattern):
            continue

        used_et[idxs_nan] = use_extra_trees(parameters, nuclei_meas)

        # Determine whether GLY and/or PRO need special treatment.
        gly_available = "GLY" in AATs_unique
        pro_available = "PRO" in AATs_unique

        gly_ambiguous = (
            {"HB", "CB", "HN"}.isdisjoint(nuclei_meas)
            and gly_available
            and pro_available
        )

        gly_only_ambiguous = (
            {"HB", "CB"}.isdisjoint(nuclei_meas)
            and gly_available
        )

        pro_only_ambiguous = (
            {"HN"}.isdisjoint(nuclei_meas)
            and pro_available
        )

        # Branch 1: both GLY and PRO are possible.
        if gly_ambiguous:
            lda = get_classifier(parameters, nuclei_meas)

            train_set_aux = train_t[nuclei_meas].copy()
            labels = train_labels.copy()

            valid = ~train_set_aux.isnull().any(axis=1)

            train_set_aux = train_set_aux.loc[valid]
            labels = labels[valid]

            if train_set_aux.empty or len(np.unique(labels)) < 2:
                raise ValueError(
                    f"Insufficient valid training classes for "
                    f"BMRB {protein}; nuclei={nuclei_meas}"
                )

            lda.fit(train_set_aux, labels)

            probs_aux = lda.predict_proba(test_missing_comb)

            probs[idxs_nan, :] = align_probabilities(
                probs_aux,
                lda.classes_,
            )

            test_labels = list(
                lda.predict(test_missing_comb)
            )

        # Branch 2: GLY is possible, but PRO is excluded.
        elif gly_only_ambiguous:
            lda = get_classifier(parameters, nuclei_meas)

            train_set_aux = train_t.loc[
                train_t["amino"] != "PRO",
                nuclei_meas,
            ].copy()

            labels = train_labels[
                train_labels != "PRO"
            ].copy()

            valid = ~train_set_aux.isnull().any(axis=1)

            train_set_aux = train_set_aux.loc[valid]
            labels = labels[valid]

            if train_set_aux.empty or len(np.unique(labels)) < 2:
                raise ValueError(
                    f"Insufficient valid training classes for "
                    f"BMRB {protein}; nuclei={nuclei_meas}"
                )

            lda.fit(train_set_aux, labels)

            probs_aux = lda.predict_proba(test_missing_comb)

            probs[idxs_nan, :] = align_probabilities(
                probs_aux,
                lda.classes_,
            )

            test_labels = list(
                lda.predict(test_missing_comb)
            )

        # Branch 3: PRO is possible, but GLY is excluded.
        elif pro_only_ambiguous:
            lda = get_classifier(parameters, nuclei_meas)

            train_set_aux = train_t.loc[
                train_t["amino"] != "GLY",
                nuclei_meas,
            ].copy()

            labels = train_labels[
                train_labels != "GLY"
            ].copy()

            valid = ~train_set_aux.isnull().any(axis=1)

            train_set_aux = train_set_aux.loc[valid]
            labels = labels[valid]

            if train_set_aux.empty or len(np.unique(labels)) < 2:
                raise ValueError(
                    f"Insufficient valid training classes for "
                    f"BMRB {protein}; nuclei={nuclei_meas}"
                )

            lda.fit(train_set_aux, labels)

            probs_aux = lda.predict_proba(test_missing_comb)

            probs[idxs_nan, :] = align_probabilities(
                probs_aux,
                lda.classes_,
            )

            test_labels = list(
                lda.predict(test_missing_comb)
            )

        # Branch 4: classify the other amino acids, excluding GLY and PRO.
        else:
            lda = get_classifier(parameters, nuclei_meas)

            train_set_aux = train_t.query(
                'amino != "PRO" and amino != "GLY"'
            )[nuclei_meas].copy()

            labels = train_labels[
                np.logical_and(
                    train_labels != "PRO",
                    train_labels != "GLY",
                )
            ].copy()

            valid = ~train_set_aux.isnull().any(axis=1)

            train_set_aux = train_set_aux.loc[valid]
            labels = labels[valid]

            if train_set_aux.empty or len(np.unique(labels)) < 2:
                raise ValueError(
                    f"Insufficient valid training classes for "
                    f"BMRB {protein}; nuclei={nuclei_meas}"
                )

            lda.fit(train_set_aux, labels)

            probs_aux = lda.predict_proba(test_missing_comb)

            # This replaces all manual np.c_ insertions.
            probs[idxs_nan, :] = align_probabilities(
                probs_aux,
                lda.classes_,
            )

            test_labels = list(
                lda.predict(test_missing_comb)
            )

        # Assign predictions to the correct test rows.
        row_indices = np.flatnonzero(idxs_nan)

        if len(test_labels) != len(row_indices):
            raise ValueError(
                f"Prediction count mismatch for BMRB {protein}: "
                f"{len(test_labels)} predictions for "
                f"{len(row_indices)} test residues"
            )

        for row_idx, label in zip(row_indices, test_labels):
            test_lab[row_idx] = label

    # Create output dataframes with consistent row and class ordering.
    df_probs = pd.DataFrame(
        probs,
        index=test_t.index,
        columns=AATs_unique,
    )

    df_protein = pd.DataFrame(
        {
            "real labels": check_labels,
            "classified as": test_lab,
            "LDA score": df_probs.max(axis=1).to_numpy(),
        },
        index=test_t.index,
    )

    content["train"] = train_t
    content["test"] = test_t
    content["check_labels"] = check_labels
    content["train_labels"] = train_labels
    content["lda_labels"] = df_protein
    content["lda_probs"] = df_probs
    content["used_et"] = used_et

    content["lda_accuracy"] = accuracy(
        check_labels, test_lab
    )

    # Save ordinary LDA results.
    if algo in ["lda", "lda_bootstrap"]:
        df_protein.to_csv(
            Path(output_dir)
            / f"labels_{protein}_{algo}_{note}.csv"
        )

        df_probs.to_csv(
            Path(output_dir)
            / f"probabilities_{protein}_{algo}_{note}.csv"
        )

    # Apply the filtering algorithm.
    if algo in ["filtered", "filtered_bootstrap"]:
        fltr = [
            i
            for i, aa in enumerate(AATs_unique)
            if AATs_count.get(aa, 0) / len(check_labels) > 0.01
        ]

        protein_filter[protein] = (
            probs[:, fltr].max(axis=1) == 1
        ) | fixed_et_rows(parameters, used_et, probs)

        filter_p = pd.DataFrame(
            protein_filter[protein],
            index=test_t.index,
            columns=[f"{protein}"],
        )

        content["filter"] = filter_p

        content["count_rest"] = AATs_count - Counter(
            [
                label
                for i, label in enumerate(test_lab)
                if protein_filter[protein][i]
            ]
        )

    return content



################################### LDA CLASSIFY ###############################


def lda_classification_classify(
    parameters, excel_protein, scanlist, fasta_path, output_dir
):
    with open(fasta_path, "r") as f:
        fasta = "".join(line.strip() for line in f)

    content = {}
    algo = parameters["algo"]
    note = lda_note(parameters)
    reference_temperature = parameters["reference_temperature"]
    temperature_correction = parameters["temperature_correction"]
    reference_ph = parameters["reference_ph"]
    ph_correction = parameters["ph_correction"]
    sequences = {}

    names = [
        "H", "HA", "HA2", "HA3", "HB", "HB1", "HB2", "HB3",
        "CA", "CB", "C", "CO", "N",
        "amino", "protein", "temperature_K",
        "seq_id", "pH", "ionic_strength",
    ]

    test_data = pd.read_excel(excel_protein, engine="openpyxl")
    TableB = pd.DataFrame(columns=names)

    # ---------------------------------------------------------
    # 1. Download BMRB entries and preserve their temperatures.
    # ---------------------------------------------------------
    for k, l in enumerate(scanlist):
        sys.stdout.write(
            f"\nScanning data from BMRB: {k + 1} of {len(scanlist)}"
        )
        sys.stdout.flush()

        ent = fetch_bmrb_entry(l)

        spectral_peaks = ent.get_saveframes_by_category(
            "assigned_chemical_shifts"
        )

        if not spectral_peaks:
            continue

        shift_frame = spectral_peaks[0]

        temperature_K = get_temperature_for_shift_list(
            ent,
            shift_frame,
        )
        pH = get_ph_for_shift_list(ent, shift_frame)
        ionic_strength = get_ionic_strength_for_shift_list(
            ent, shift_frame
        )
        sequences[l] = get_sequence_for_shift_list(ent, shift_frame)

        TableC = pd.DataFrame(columns=names)

        w = shift_frame["_Atom_chem_shift.Seq_ID"]
        x = shift_frame["_Atom_chem_shift.Comp_ID"]
        y = shift_frame["_Atom_chem_shift.Atom_ID"]
        z = shift_frame["_Atom_chem_shift.Val"]

        if not (len(w) == len(x) == len(y) == len(z)):
            raise ValueError(
                f"Inconsistent chemical-shift arrays for BMRB {l}: "
                f"Seq_ID={len(w)}, Comp_ID={len(x)}, "
                f"Atom_ID={len(y)}, Val={len(z)}"
            )

        if not w:
            continue

        wp = w[0]
        n = 0

        for i, seq_id in enumerate(w):
            if seq_id != wp:
                n += 1
                wp = seq_id

            TableC.at[n, "protein"] = l
            TableC.at[n, "amino"] = x[i]
            TableC.at[n, "temperature_K"] = temperature_K
            TableC.at[n, "seq_id"] = seq_id
            TableC.at[n, "pH"] = pH
            TableC.at[n, "ionic_strength"] = ionic_strength

            atom_name = str(y[i]).strip()
            shift_value = pd.to_numeric(z[i], errors="coerce")

            if pd.isna(shift_value):
                continue

            if atom_name in names:
                TableC.at[n, atom_name] = float(shift_value)

        TableB = pd.concat(
            [TableB, TableC],
            ignore_index=True,
        )

    # ---------------------------------------------------------
    # 2. Assemble residue-level features and preserve metadata.
    # ---------------------------------------------------------
    TableA = pd.DataFrame(index=TableB.index)

    TableA["amino"] = TableB["amino"]
    TableA["protein"] = TableB["protein"]
    TableA["temperature_K"] = pd.to_numeric(
        TableB["temperature_K"],
        errors="coerce",
    )
    TableA["seq_id"] = TableB["seq_id"]
    TableA["pH"] = pd.to_numeric(
        TableB["pH"],
        errors="coerce",
    )
    TableA["ionic_strength"] = pd.to_numeric(
        TableB["ionic_strength"],
        errors="coerce",
    )

    # Average beta-proton shifts.
    hb_columns = ["HB", "HB1", "HB2", "HB3"]
    TableA["HB"] = TableB[hb_columns].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)

    TableA["HN"] = pd.to_numeric(
        TableB["H"], errors="coerce"
    )
    TableA["CA"] = pd.to_numeric(
        TableB["CA"], errors="coerce"
    )
    TableA["CB"] = pd.to_numeric(
        TableB["CB"], errors="coerce"
    )
    TableA["N"] = pd.to_numeric(
        TableB["N"], errors="coerce"
    )

    # BMRB atom name C is the carbonyl carbon.
    TableA["CO"] = pd.to_numeric(
        TableB["C"], errors="coerce"
    )

    # Average available alpha-proton shifts.
    ha_columns = ["HA", "HA2", "HA3"]
    TableA["HA"] = TableB[ha_columns].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)

    # ---------------------------------------------------------
    # 3. Correct training shifts before feature scaling.
    # ---------------------------------------------------------
    if temperature_correction:
        TableA = correct_temperature_shifts(
            TableA,
            reference_temperature=reference_temperature,
        )

    if ph_correction:
        TableA = correct_ph_shifts(
            TableA,
            sequences,
            reference_ph=reference_ph,
            skip_termini=parameters.get("ph_skip_termini", False),
            skip_neutral=parameters.get("ph_skip_neutral", False),
        )

    # The temperature metadata is deliberately not a classifier feature.
    AAT_dict = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
        "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
        "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
        "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
        "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }

    # Amino-acid types expected in the FASTA sequence.
    fasta = fasta.upper()
    invalid_residues = sorted(set(fasta) - set(AAT_dict))

    if invalid_residues:
        raise ValueError(
            f"FASTA contains unsupported amino-acid symbols: "
            f"{invalid_residues}"
        )

    AATs_unique = sorted({AAT_dict[aa] for aa in set(fasta)})
    AATs_count = Counter(AAT_dict[aa] for aa in fasta)

    # Test data: SSN identifies the residue; amino is a label column
    # if present in the supplied spreadsheet.
    SSN = test_data.iloc[:, 0]
    check_labels = pd.Series(
        [None] * len(test_data),
        index=test_data.index,
    )

    test_set = test_data.drop(columns=["SSN"]).copy()

    if "amino" not in test_set.columns:
        raise ValueError(
            "The Excel test data must contain an 'amino' column "
            "for compatibility with the current classification pipeline."
        )

    nuclei = [
        col for col in test_set.columns
        if col not in {"amino", "protein", "temperature_K"}
    ]

    # Only use the selected test nuclei as classifier features.
    missing_features = [
        col for col in nuclei if col not in TableA.columns
    ]
    if missing_features:
        raise ValueError(
            f"Test features are not available in the BMRB training data: "
            f"{missing_features}"
        )

    # Sort training data by amino-acid type, matching the original workflow.
    train_data = TableA[
        nuclei + ["amino", "protein"]
    ].sort_values("amino", ascending=True).copy()

    # Remove training classes absent from the FASTA sequence.
    AATs_missing = list(set(AAT_dict.values()) - set(AATs_unique))

    for AAT in AATs_missing:
        train_data = train_data.drop(
            train_data.query("amino == @AAT").index
        )

    # ---------------------------------------------------------
    # 4. Scale training and test features together.
    # ---------------------------------------------------------
    # IMPORTANT: this retains the original scaling formula.
    # It assumes the Excel test features are already on the same
    # reference-temperature basis as the corrected training features.
    for_scaling = pd.concat(
        [test_set, train_data],
        ignore_index=True,
    )

    for col in nuclei:
        values = pd.to_numeric(
            for_scaling[col], errors="coerce"
        )
        mean = values.mean()
        std = values.std(ddof=0)

        if pd.isna(std) or std == 0:
            for_scaling[col] = values - mean
        else:
            for_scaling[col] = (
                values - mean
            ) / np.sqrt(std)

    n_test = len(test_data)
    test_t = for_scaling.iloc[:n_test].copy()
    train_t = for_scaling.iloc[n_test:].copy()

    # ---------------------------------------------------------
    # 5. Balance training data using bootstrap resampling.
    # ---------------------------------------------------------
    train_labels = train_t["amino"].values
    balanced_data = []

    for amino_acid in train_t["amino"].unique():
        class_data = train_t[
            train_t["amino"] == amino_acid
        ]

        n_samples = train_t["amino"].value_counts().max()

        balanced_class_data = resample(
            class_data,
            replace=True,
            n_samples=n_samples,
            random_state=parameters["seed"],
        )
        balanced_data.append(balanced_class_data)

    balanced_train_data = (
        pd.concat(balanced_data, ignore_index=True)
        if balanced_data
        else train_t.copy()
    )

    # Drop residues that have no measured classifier features.
    test_t = test_t.dropna(subset=nuclei, how="all")
    train_t = train_t.dropna(subset=nuclei, how="all")

    if algo in ["lda_bootstrap", "filtered_bootstrap"]:
        train_t = balanced_train_data.dropna(
            subset=nuclei, how="all"
        )

    train_labels = train_t["amino"].values

    # Align the unknown test labels to the rows retained for prediction.
    check_labels = check_labels.loc[test_t.index].values

    test_lab = [None for _ in range(test_t.shape[0])]
    probs = np.ndarray(
        shape=(len(test_lab), len(AATs_unique))
    )

    test_nan = np.array(test_t[nuclei].isnull())
    combs_nan = np.unique(test_nan, axis=0)

    # Rows classified by ExtraTrees (classifier = et_lda); never passed to SW.
    used_et = np.zeros(len(test_t), dtype=bool)

    if algo == "sw":
        content["train"] = train_t
        content["test"] = test_t
        content["check_labels"] = check_labels
        content["train_labels"] = train_labels
        content["count_rest"] = AATs_count
        return content

    # Get column positions of GLY and PRO.
    if "GLY" in AATs_unique:
        G_idx = AATs_unique.index("GLY")
    if "PRO" in AATs_unique:
        P_idx = AATs_unique.index("PRO")

    # ---------------------------------------------------------
    # 6. Classification for each pattern of missing nuclei.
    # ---------------------------------------------------------
    for i in combs_nan:
        idxs_nan = np.all(test_nan == i, axis=1)
        nuclei_meas = list(compress(nuclei, ~i))
        test_missing_comb = test_t.loc[idxs_nan, nuclei_meas]

        # All features are missing.
        if np.all(i):
            a = np.empty(
                (idxs_nan.sum(), probs.shape[1])
            )
            a[:] = np.nan
            probs[idxs_nan, :] = a
            continue

        used_et[idxs_nan] = use_extra_trees(parameters, nuclei_meas)

        # Residues that could be GLY or PRO.
        if (
            {"HB", "CB", "HN"}.isdisjoint(nuclei_meas)
            and all(x in AATs_unique for x in ["GLY", "PRO"])
        ):
            lda = get_classifier(parameters, nuclei_meas)
            idxs = train_t[nuclei_meas].isnull().any(axis=1)
            train_set_aux = train_t.loc[
                ~idxs, nuclei_meas
            ].copy()
            labels = train_labels[~idxs].copy()

            lda.fit(train_set_aux, labels)
            lda.means_ = 1

            probs[idxs_nan, :] = lda.predict_proba(
                test_missing_comb
            )
            test_labels = list(
                lda.predict(test_missing_comb)
            )

            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Residues that could be GLY.
        elif (
            {"HB", "CB"}.isdisjoint(nuclei_meas)
            and "GLY" in AATs_unique
        ):
            lda = get_classifier(parameters, nuclei_meas)
            train_set_aux = train_t.loc[
                train_t["amino"] != "PRO", nuclei_meas
            ].copy()
            labels = train_labels[
                train_labels != "PRO"
            ].copy()

            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux.loc[~idxs]
            labels = labels[~idxs]

            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))

            if "PRO" in AATs_unique:
                probs[idxs_nan, :] = np.c_[
                    probs_aux[:, :P_idx],
                    np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, P_idx:],
                ]
            else:
                probs[idxs_nan, :] = probs_aux

            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Residues that could be PRO.
        elif (
            {"HN"}.isdisjoint(nuclei_meas)
            and "PRO" in AATs_unique
        ):
            lda = get_classifier(parameters, nuclei_meas)
            train_set_aux = train_t.loc[
                train_t["amino"] != "GLY", nuclei_meas
            ].copy()
            labels = train_labels[
                train_labels != "GLY"
            ].copy()

            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux.loc[~idxs]
            labels = labels[~idxs]

            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))

            if "PRO" not in set(AATs_unique).difference(set(labels)):
                if "GLY" in AATs_unique:
                    probs[idxs_nan, :] = np.c_[
                        probs_aux[:, :G_idx],
                        np.zeros((idxs_nan.sum(), 1)),
                        probs_aux[:, G_idx:],
                    ]
                else:
                    probs[idxs_nan, :] = probs_aux
            else:
                if "GLY" in AATs_unique:
                    probs[idxs_nan, :] = np.c_[
                        probs_aux[:, :G_idx],
                        np.zeros((idxs_nan.sum(), 1)),
                        probs_aux[:, G_idx:P_idx - 1],
                        np.zeros((idxs_nan.sum(), 1)),
                        probs_aux[:, P_idx - 1:],
                    ]
                else:
                    probs[idxs_nan, :] = np.c_[
                        probs_aux[:, :P_idx],
                        np.zeros((idxs_nan.sum(), 1)),
                        probs_aux[:, P_idx:],
                    ]

            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # All other residues: exclude BOTH GLY and PRO.
        else:
            lda = get_classifier(parameters, nuclei_meas)
            train_set_aux = train_t.query(
                'amino != "PRO" and amino != "GLY"'
            )[nuclei_meas].copy()

            labels = train_labels[
                (train_labels != "PRO")
                & (train_labels != "GLY")
            ].copy()

            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux.loc[~idxs]
            labels = labels[~idxs]

            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))

            if all(x in AATs_unique for x in ["GLY", "PRO"]):
                probs[idxs_nan, :] = np.c_[
                    probs_aux[:, :G_idx],
                    np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, G_idx:P_idx - 1],
                    np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, P_idx - 1:],
                ]
            elif "PRO" in AATs_unique:
                probs[idxs_nan, :] = np.c_[
                    probs_aux[:, :P_idx],
                    np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, P_idx:],
                ]
            elif "GLY" in AATs_unique:
                probs[idxs_nan, :] = np.c_[
                    probs_aux[:, :G_idx],
                    np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, G_idx:],
                ]
            else:
                probs[idxs_nan, :] = probs_aux

            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

    # ---------------------------------------------------------
    # 7. Store and export classification results.
    # ---------------------------------------------------------
    df_probs = pd.DataFrame(
        probs,
        index=test_t.index,
        columns=AATs_unique,
    )

    df_protein = pd.DataFrame({
        "real labels": check_labels,
        "classified as": test_lab,
        "LDA score": df_probs.max(axis=1).values,
    })

    content["train"] = train_t
    content["test"] = test_t
    content["check_labels"] = check_labels
    content["train_labels"] = train_labels
    content["lda_labels"] = df_protein
    content["lda_probs"] = df_probs
    content["used_et"] = used_et

    if algo == "lda":
        df_protein.to_csv(
            Path(output_dir) / f"labels_lda_{note}.csv"
        )
        df_probs.to_csv(
            Path(output_dir) / f"probabilities_lda_{note}.csv"
        )

    if algo == "lda_bootstrap":
        df_protein.to_csv(
            Path(output_dir) / f"labels_lda_bootstrap_{note}.csv"
        )
        df_probs.to_csv(
            Path(output_dir) / f"probabilities_lda_bootstrap_{note}.csv"
        )

    if algo in ["filtered", "filtered_bootstrap"]:
        fltr = [
            i
            for i, a in enumerate(AATs_unique)
            if AATs_count.get(a, 0) / len(check_labels) > 0.01
        ]

        protein_filter = (
            probs[:, fltr].max(axis=1) == 1
        ) | fixed_et_rows(parameters, used_et, probs)
        filter_p = pd.DataFrame(protein_filter)
        content["filter"] = filter_p

        cnt_rest = Counter(
            [
                lab
                for i, lab in enumerate(test_lab)
                if protein_filter[i]
            ]
        )
        content["count_rest"] = AATs_count - cnt_rest

    return content



################################### PREPARE FOR SW #############################

def prepare_data(algo, content, sample_method, seed, eps):
    assert algo in {"filtered", "filtered_bootstrap", "sw"}

    te = content["test"]
    tr = content["train"]

    if algo in {"filtered", "filtered_bootstrap"}:
        prob = content["lda_probs"]
        fl =  content["filter"]
        fl_num = fl.to_numpy().flatten()
        prob_sw = prob.loc[~fl_num, :]
        te = te.loc[~fl_num, :]
        amino_count = Counter(te["amino"].values)
    else:
        prob_sw = None
    AATs_count = content["count_rest"]
    nuclei = te.drop(columns=["amino", "protein"]).columns
    if sample_method == "BEST":
        tr = find_best_reference(AATs_count, tr, nuclei, eps)
    elif sample_method == "ALL":
        # Every train point of the classes still to assign (W weights them by count_rest)
        tr = tr[tr["amino"].isin([amino for amino, cnt in AATs_count.items() if cnt > 0])]
    elif sample_method in {"RAND", "CENT"}:
        is_rand = sample_method == "RAND"
        samples = []
        index = 0
        for amino, group in tr.groupby("amino"):
            if is_rand:
                sampled_group = group.sample(AATs_count[amino], random_state=seed)
            else:
                centroid = group[nuclei].mean().values
                sampled_group = pd.DataFrame(
                    [list(centroid) + [amino, 0] for _ in range(AATs_count[amino])],
                    columns=nuclei + ["amino", "protein"],
                    index=range(index, index + AATs_count[amino])
                )
            samples.append(sampled_group)
            index += AATs_count[amino]

        tr = pd.concat(samples)
    return (tr, te, prob_sw)

############################ SW CLASSIFICATION #################################

def prepare_labels(train: pd.DataFrame, test: pd.DataFrame):
    train_labels = train["amino"].values
    check_labels = test["amino"].values
    test_drop = test.drop(columns=["amino", "protein"])
    nuclei = test_drop.columns.tolist()
    return (train_labels, check_labels, test_drop, nuclei)

def useful_structures(test: pd.DataFrame):
    test_lab = [None for _ in range(test.shape[0])]
    test_nan = np.array(test.isnull())
    combs_nan = np.unique(test_nan, axis=0)
    prob_matrix = np.zeros((20, test.shape[0]))
    return (test_lab, test_nan, combs_nan, prob_matrix)

def classify_in_combination(
    train, train_labels, test, check_labels, nuclei, test_lab, test_nan, prob_matrix,
    comb, parameters, params_comb, count_rest, class_mass=None
):
    n_projections = parameters["n_projections"]
    seed = parameters["seed"]
    k = parameters["k"]
    device = parameters["device"]
    with_bins = parameters["with_bins"]
    bins_backend = parameters["bins_backend"]
    vote_type = parameters["vote_type"]
    scale = parameters["scale"]
    kappa1 = parameters["kappa1"]
    kappa2 = parameters["kappa2"]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if k == 0:
        k = None

    idxs_nan = np.all(test_nan == comb, axis=1)
    nuclei_meas = list(compress(nuclei, ~comb))
    test_missing_comb = test[idxs_nan][nuclei_meas]
    check_comb_labels = check_labels[idxs_nan]
    if np.all(comb):
        print("residues with all NaN values")
        return None
    train_set_aux = train[nuclei_meas].copy()
    train_comb_labels = train_labels.copy()
    idxs = train_set_aux.isnull().any(axis=1)
    train_set_aux = train_set_aux[~idxs]
    train_comb_labels = train_comb_labels[~idxs]
    test_comb = test_missing_comb.values
    train_comb = train_set_aux.values
    x1_torch = torch.tensor(train_comb).to(device=device).requires_grad_(True)
    x2_torch = torch.tensor(test_comb).to(device=device)
    if "CA" in nuclei_meas:
        dimPRO=nuclei_meas.index("CA")
        n_specPRO = 1
    else:
        dimPRO=None
        n_specPRO = 0
    if "HN" in nuclei_meas:
        dimGLY=nuclei_meas.index("HN")
        n_specGLY = 1
    else:
        dimGLY=None
        n_specGLY = 0

    AATs_unique = list(set(train_comb_labels))
    AATs_unique.sort()

    if parameters["transport"] == "w":
        test_labels, total_transport_classes = classify_w(
            train_comb, test_comb, train_comb_labels, classes=ALL_POSSIBLE, kappa1=kappa1,
            kappa2=kappa2, with_bins=with_bins, metric=parameters.get("w_metric", "L2"),
            class_mass=class_mass
        )
    elif vote_type == "NORMAL":
        test_labels, total_transport_classes = classify_sw(
            x1_torch, x2_torch, train_comb_labels, classes=ALL_POSSIBLE,
            n_projections=n_projections, seed=gen, device=device, scale=scale, k=k, kappa1=kappa1,
            kappa2=kappa2, with_bins=with_bins, bins_backend=bins_backend
        )
    elif vote_type == "SPECIAL":
        test_labels, total_transport_classes = classify_sw_spec(
            x1_torch, x2_torch, train_comb_labels, count_rest, classes = ALL_POSSIBLE,
            dimPRO=dimPRO, dimGLY=dimGLY, n_projections=n_projections, n_specPRO=n_specPRO,
            n_specGLY=n_specGLY, parameters_amino=params_comb, seed=gen, device=device,
            scale=scale, k=k, kappa1=kappa1, kappa2=kappa2, with_bins=with_bins,
            bins_backend=bins_backend
        )
    else:
        assert False, "Only NORMAL or SPECIAL"

    # Residues assigned here are no longer available to the next combinations
    count_rest.subtract(Counter(test_labels))

    li = 0
    for idx, val in enumerate(idxs_nan):
        if val and test_labels:
            test_lab[idx] = test_labels.pop(0)
            prob_matrix[:, idx] = total_transport_classes[:, li]
            li +=1
    return (test_lab, prob_matrix)

def validate(i, train, test, parameters, parameters_amino, param_str, output_dir, count_rest):
    train_labels, check_labels, test_drop, nuclei = prepare_labels(train, test)
    nuclei = parameters["nuclei"]
    useful_structs = useful_structures(test_drop)
    test_lab, test_nan, combs_nan, prob_matrix = useful_structs
    # Copy, shared by all combinations and decreased as residues get assigned
    count_rest = Counter(count_rest)
    # W with every train point (sample_method = "ALL"): class masses from the expected
    # composition of all residues left, the same for every combination
    class_mass = None
    if parameters["transport"] == "w" and parameters["sample_method"] == "ALL":
        rest = {lab: cnt for lab, cnt in count_rest.items() if cnt > 0}
        class_mass = {lab: cnt / sum(rest.values()) for lab, cnt in rest.items()}

    for k, comb in enumerate(combs_nan):
        print(f"{datetime.now().time()} Task {i} in comb {k+1} of {len(combs_nan)}.", flush=True)
        result = classify_in_combination(train, train_labels, test, check_labels, nuclei, test_lab,\
                             test_nan, prob_matrix, comb, parameters, parameters_amino, count_rest,
                             class_mass)
        if result is None:
            continue

        test_lab, prob_matrix = result


    df_prob = pd.DataFrame(
        normalize_matrix_col(prob_matrix, 0), index = ALL_POSSIBLE, columns=check_labels
    )
    df_protein = pd.DataFrame({"real labels": check_labels, "classified as": test_lab, "score SW": df_prob.T.max(axis=1).values})

    if parameters["algo"] == "sw":
        print(f"Labels saved in {output_dir}labels_"+param_str+".csv")
        df_protein.to_csv(Path(output_dir) / f"labels_{param_str}.csv")
        print(f"Probabilities saved in {output_dir}probabilities_"+param_str+".csv")
        df_prob.to_csv(Path(output_dir) / f"probabilities_{param_str}.csv")

    return {
        "sw_labels":df_protein,
        "sw_probs":df_prob,
        "sw_accuracy": accuracy(check_labels, test_lab)
    }

######################################## COMBINE #######################################
def assign_to_composition(scores, classes, count_rest, slack=1.0):
    """
    Labels maximising the summed scores (residues x classes), when every class has as many
    slots as expected from count_rest (scaled to the number of residues) times slack.
    """
    n_res = scores.shape[0]
    if n_res == 0:
        return []
    counts = np.array([max(count_rest.get(c, 0), 0) for c in classes], dtype=np.float64)
    slots = np.round(counts / max(counts.sum(), 1) * n_res).astype(int)
    slots = np.ceil(slots * slack + (slack > 1)).astype(int)
    cols = np.repeat(np.arange(len(classes)), slots)
    if len(cols) < n_res:
        # Never fewer slots than residues
        cols = np.concatenate([cols, np.repeat(np.arange(len(classes)), n_res)])
    rows, picked = linear_sum_assignment(-scores[:, cols])
    labels = [None] * n_res
    for r, c in zip(rows, picked):
        labels[r] = classes[cols[c]]
    return labels

def combine_by_scores(prob_lda, prob_sw, count_rest, parameters):
    """
    Labels and probabilities of the residues left after filtering, from the classifier
    (LDA/ET, residues x classes) and transport (SW/W, classes x residues) probabilities.
    combine = "mix": argmax of w * classifier + (1 - w) * transport.
    combine = "assign": residues assigned to the slots given by count_rest, maximising the
    sum of w * log(classifier + 1e-4) + (1 - w) * log(transport + floor).
    """
    assert parameters["combine"] in {"mix", "assign"}, "combine: tree, mix or assign"
    classes = list(prob_sw.index)
    p_lda = prob_lda.reindex(columns=classes, fill_value=0.0).to_numpy(dtype=np.float64)
    p_tr = prob_sw.T.to_numpy(dtype=np.float64)
    weight = parameters["combine_weight"]

    if parameters["combine"] == "mix":
        probs = weight * p_lda + (1 - weight) * p_tr
        return [classes[j] for j in probs.argmax(axis=1)], probs

    scores = weight * np.log(p_lda + 1e-4) + \
        (1 - weight) * np.log(p_tr + parameters.get("combine_floor", 0.1))
    # Pooled probabilities, normalised LDA^w * transport^(1-w)
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = assign_to_composition(
        scores, classes, count_rest, parameters.get("combine_slack", 1.0)
    )
    return labels, probs

def combine_methods(output_dir, content_lda, content_sw, param_str, parameters=None):
    # Extract relevant data
    prob_sw = content_sw["sw_probs"]
    sw_labels = content_sw["sw_labels"]
    fl = content_lda["filter"].to_numpy().flatten()

    lda_labels = content_lda["lda_labels"].loc[~fl, :]
    lda_labels_new = content_lda["lda_labels"].loc[fl, :]
    prob_lda = content_lda["lda_probs"].loc[~fl, :]
    probs_combined = content_lda["lda_probs"].copy()

    # Ensure all amino acids (AA) from prob_sw are present as columns in probs_combined
    desired_columns = list(prob_sw.index)

    # Reindex columns to match desired AA order, filling missing ones with 0.0
    probs_combined = probs_combined.reindex(columns=desired_columns, fill_value=0.0)

    # Count real label occurrences in LDA labels
    # AATs_count = Counter(lda_labels["real labels"].values)
    # Expected class sizes of the residues left after LDA filtering, from the sequence
    # (real labels are unknown when classifying, and leak the answer when validating)
    AATs_count = Counter({lab: cnt for lab, cnt in content_lda["count_rest"].items() if cnt > 0})
    n_rest = max(sum(AATs_count.values()), 1)
    best_method  = None
    methods = []
    # combine = "tree" (default): the rule below; "mix" / "assign": combine_by_scores
    rule = parameters.get("combine", "tree") if parameters else "tree"
    rest_labels, rest_probs = None, None
    if rule != "tree":
        rest_labels, rest_probs = combine_by_scores(prob_lda, prob_sw, AATs_count, parameters)
    # Adjust SW label classification based on LDA and SW scores (probabilities)
    for i in range(len(sw_labels.index)):
        if rest_probs is not None:
            sw_labels.at[i, "classified as"] = str(rest_labels[i])
            sw_labels.at[i, "LDA score"] = prob_lda.idxmax(axis=1).iloc[i]
            methods.append(rule.upper())
            continue
        best_label = sw_labels["classified as"].iloc[i]
        best_method = 'SW'
        if prob_lda.max(axis=1).iloc[i] >= 0.55:
            if AATs_count[prob_lda.idxmax(axis=1).iloc[i]]/n_rest >= 0.01:
                best_label = prob_lda.idxmax(axis=1).iloc[i]
                best_method = 'LDA'
            if prob_sw.max(axis=0).iloc[i] >= 0.8 and prob_sw.idxmax(axis=0).iloc[i] != best_label:
                best_label = prob_sw.idxmax(axis=0).iloc[i]
                best_method = 'SW'

        # Handle empty or invalid labels
        if best_label is None or best_label == np.nan or str(best_label).strip() == "None":
            best_label = prob_lda.idxmax(axis=1).iloc[i]
            best_method = 'LDA'
        sw_labels.at[i, "classified as"] = str(best_label)
        sw_labels.at[i, "LDA score"] = prob_lda.idxmax(axis=1).iloc[i]
        methods.append(best_method)

    # Update index to match unfiltered entries
    sw_labels.index = [i for i, b in enumerate(fl) if not b]
    methods_zipped = zip([i for i, b in enumerate(fl) if not b], methods)

    for i, (j, lab) in enumerate(methods_zipped):
        if lab == 'LDA':
            continue
        else:
            if rest_probs is None:
                prob_values = prob_sw.iloc[:, i].to_numpy()
            else:
                prob_values = rest_probs[i]
            probs_combined.iloc[j, :] = prob_values

    probs_combined = probs_combined.where(probs_combined > 0.01, 0.0)

    # Combine updated SW and LDA labels
    combined = pd.concat([sw_labels, lda_labels_new], copy=True)
    combined.sort_index(inplace=True)
    combined = combined.replace({np.nan: None})

    # Save output
    combined.to_csv(Path(output_dir) / f"combined_labels_{param_str}.csv")
    probs_combined.to_csv(Path(output_dir) / f"combined_probabilities_{param_str}.csv")

    # Compute accuracy if real labels exist
    acc = accuracy(combined["real labels"].values, combined["classified as"].values)
    if all([not bool(x) for x in combined["real labels"].values]):
        return {"combined_labels": combined["classified as"]}
    return {"combined_labels":combined, "combined_accuracy" : acc}

# # 0.895
# Decision-making algorithm based on the best decision tree by ML approach
#                           (DecisionTreeClassifier(max_depth=4, random_state=42))
# def choose_label(score_SW, score_LDA, class_size_SW, class_size_LDA):
#     if score_SW <= 0.93:
#         if score_LDA <= 0.60:
#             if score_SW <= 0.58:
#                 return 1 if class_size_LDA <= 0.04 else 0
#             else:
#                 return 1
#         else:
#             if class_size_SW <= 0.20:
#                 return 0
#             else:
#                 return 1 if score_LDA <= 0.76 else 0
#     else:
#         if score_SW <= 1.00:
#             if score_LDA <= 0.84:
#                 return 0 if class_size_SW <= 0.07 else 1
#             else:
#                 return 0
#         else:
#             if class_size_LDA <= 0.14:
#                 return 1
#             elif class_size_LDA <= 0.15:
#                 return 0
#             else:
#                 return 1
#
# def combine_methods(protein, output_dir, content_lda, content_sw, param_str):
#     prob_sw = content_sw["sw_probs"]
#     content_filter = content_lda["filter"]
#     sw_labels = content_sw["sw_labels"]
#     fl = content_filter.to_numpy().flatten()

#     # Split LDA labels
#     lda_labels = content_lda["lda_labels"].loc[~fl, :]
#     lda_labels_new = content_lda["lda_labels"].loc[fl, :]
#     prob_lda = content_lda["lda_probs"].loc[~fl, :]

#     # Compute class size dictionaries
#     AATs_count_lda = Counter(lda_labels["real labels"].values)
#     AATs_count_sw = Counter(sw_labels["real labels"].values)
#     total_lda = len(lda_labels)
#     total_sw = len(sw_labels)

#     sw_labels = sw_labels.copy()
#     sw_labels["score_SW"] = prob_sw.max(axis=0).values
#     sw_labels["pred_SW"] = prob_sw.idxmax(axis=0).values
#     sw_labels["score_LDA"] = prob_lda.max(axis=1).values
#     sw_labels["pred_LDA"] = prob_lda.idxmax(axis=1).values

#     # Class sizes
#     sw_labels["class_size_SW"] = sw_labels["real labels"].map(lambda x: AATs_count_sw.get(x, 0) / total_sw if total_sw else 0)
#     sw_labels["class_size_LDA"] = sw_labels["real labels"].map(lambda x: AATs_count_lda.get(x, 0) / total_lda if total_lda else 0)

#     # Apply decision tree logic
#     chosen_labels = []
#     for i, row in sw_labels.iterrows():
#         label = choose_label(
#             row["score_SW"],
#             row["score_LDA"],
#             row["class_size_SW"],
#             row["class_size_LDA"]
#         )
#         chosen_label = row["pred_SW"] if label == 1 else row["pred_LDA"]
#         chosen_labels.append(str(chosen_label))

#     sw_labels["classified as"] = chosen_labels
#     sw_labels.index = [i for i, b in enumerate(fl) if not b]

#     combined = pd.concat([sw_labels, lda_labels_new], copy=True)
#     combined.sort_index(inplace=True)
#     combined = combined.replace({np.nan: None})

#     if protein:
#         combined.to_csv(output_dir + f"{protein}_combined_labels_" + param_str + ".csv", index=False)
#     else:
#         combined.to_csv(output_dir + f"combined_labels_{param_str}.csv", index=False)

#     if all([not bool(x) for x in combined["real labels"].values]):
#         return {"combined_labels": combined["classified as"]}

#     acc = accuracy(combined["real labels"].values, combined["classified as"].values)
#     return {
#         "combined_labels": combined,
#         "combined_accuracy": acc
#     }
