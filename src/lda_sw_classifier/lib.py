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
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import resample
from pathlib import Path

warnings.filterwarnings("ignore")
from masserstein import NMRSpectrum, estimate_proportions # pylint: disable=wrong-import-position


ALL_POSSIBLE = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]

NUM_COLORS = 20

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
        "note": config["parameters"]["note"]
    }
    return parameters

def read_bmrb_id(bmrbID_path):
    with open(bmrbID_path, "r") as f:
        scanlist = [line.rstrip("\n") for line in f]
    while scanlist[-1] == "":
        scanlist.pop(-1)
    return scanlist
############################## SW CLASSIFIER ###################################
def normalize_matrix_col(matrix):
    """
    Function normalizing columns.
    """
    col_sums = np.sum(matrix, axis=0)
    col_sums[col_sums == 0] = 1
    matrix = matrix/col_sums
    return matrix

def normalize_matrix_rows(matrix):
    """
    Function normalising rows.
    """
    row_sums = np.sum(matrix, axis=1)
    row_sums[row_sums == 0] = 1
    matrix = matrix/row_sums[:, np.newaxis]
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
    X_t_dim0, scale, kappa1=2.1, kappa2=2.1, k = None, with_bins = False
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
            emd = bins_transport(
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
    n_projections, n_specPRO, n_specGLY, s_labels, t_labels, classes, X_s_projections,
    X_t_projections, X_s_dim0, X_t_dim0, parameters_amino, scale, kappa1=2.1,
    kappa2=2.1, k = None, with_bins = False
):
    # Dictionary with occurrences in the test set
    t_labels = list(t_labels)
    lab_dict = {lab:t_labels.count(lab) for lab in set(t_labels)}
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
            emd = bins_transport(
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
    k = None, with_bins = False, printing=False
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
        X_t.shape[0], scale, kappa1=kappa1, kappa2=kappa2, k = k, with_bins = with_bins
    )

def classify_sw_spec(
    X_s, X_t, s_labels, t_labels, classes, n_projections, dimPRO, dimGLY, n_specPRO, n_specGLY,
    parameters_amino, seed, device, scale, kappa1=2.1, kappa2=2.1, k = None, with_bins = False,
    printing=False
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
        n_projections, n_specPRO, n_specGLY, s_labels, t_labels,
        classes, X_s_projections, X_t_projections, X_s.shape[0], X_t.shape[0], parameters_amino,
        scale, kappa1=kappa1, kappa2=kappa2, k = k, with_bins = with_bins
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

################################### LDA ########################################
def lda_classification_validate(parameters, protein, scanlist, output_dir) :
    content = {}
    algo = parameters["algo"]
    names = ["H", "HB", "HB1", "HB2", "HB3", "CA", "CB", "C", "CO", "N", "amino", "protein"]
    TableB = pd.DataFrame(columns=names)

    for k, l in enumerate(scanlist):
        waiter = True
        sys.stdout.write(f"\nScanning data from BMRB: {k + 1} of {len(scanlist)}")
        sys.stdout.flush()

        while waiter:
            try:
                ent = pynmrstar.Entry.from_database(l)
                waiter = False
            except (ConnectionError, OSError) as _:
                print("\nConnection error on: " + str(l))
                sleep(2)
                waiter = True
        spectral_peaks = ent.get_saveframes_by_category("assigned_chemical_shifts")

        if len(spectral_peaks) > 0:
            TableC = pd.DataFrame(columns=names)
            w = spectral_peaks[0]["_Atom_chem_shift.Seq_ID"]
            x = spectral_peaks[0]["_Atom_chem_shift.Comp_ID"]
            y = spectral_peaks[0]["_Atom_chem_shift.Atom_ID"]
            z = spectral_peaks[0]["_Atom_chem_shift.Val"]
            wp = w[0]
            n = 0

            for i, j in enumerate(w):
                if j != wp:
                    n = n + 1
                    wp = w[i]
                TableC.at[n, "protein"] = l
                TableC.at[n, "amino"] = x[i]
                TableC.at[n, y[i]] = float(z[i])
            TableB = pd.concat([TableB, TableC], ignore_index=True)
    print("\n")

    TableA = pd.DataFrame(
        columns=["amino", "protein", "HN", "N", "CO", "HA", "HB", "CA", "CB"]
    ).astype("float")
    try:
        TableA["HB"] = TableB.loc[:, "HB":"HB3"].astype(float).mean(axis=1).astype(float)
    except KeyError:
        TableA["HB"] = TableB.loc[:, "HB"].astype(float)

    TableA["HN"] = TableB.loc[:, "H"].astype(float)
    TableA["CA"] = TableB.loc[:, "CA"].astype(float)
    TableA["CB"] = TableB.loc[:, "CB"].astype(float)
    TableA["N"] = TableB.loc[:, "N"].astype(float)
    TableA["CO"] = TableB.loc[:, "C"].astype(float)

    try:
        TableA["HA"] = TableB.loc[:, ["HA", "HA2", "HA3"]].astype(float).mean(axis=1)
    except KeyError:
        try:
            TableA["HA"] = TableB.loc[:, "HA"].astype(float)
        except KeyError:
            pass


    TableA["amino"] = TableB["amino"]
    TableA["protein"] = TableB["protein"]

    AAT_dict = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
                "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
                "Y": "TYR", "V": "VAL"}

    nuclei = ["HN", "N", "CO", "CA", "CB", "HA", "HB"]
    train_data = pd.DataFrame(TableA, columns=nuclei+["amino", "protein"])

    protein_filter = {}
    considered = train_data[train_data["protein"].isin(scanlist)]

    check_labels = considered.loc[considered["protein"] == protein, "amino"].values
    AATs_unique = list(set(check_labels))

    AATs_count = Counter(i for i in check_labels)

    AATs_unique.sort()

    AATs_missing = list(set(AAT_dict.values()) - set(AATs_unique))
    data_considered = considered.copy()
    for AAT in AATs_missing: # pylint: disable=unused-variable
        data_considered = data_considered.copy().drop(data_considered.query("amino==@AAT").index)

    train_t = data_considered.loc[data_considered["protein"] != protein]
    test_t = data_considered.loc[data_considered["protein"] == protein]

    for_scaling = pd.concat([test_t, train_t], ignore_index=True)
    for k in for_scaling:
        if k in {"amino", "protein"}:
            break
        for_scaling[k] = (
            for_scaling[k] - np.mean(for_scaling[k])) / (np.sqrt(np.std(for_scaling[k]))
        )

    test_t = for_scaling.iloc[:len(test_t), :]
    train_t = for_scaling.iloc[len(test_t):, :]

    # Balancing data using bootstrap resampling
    train_labels = train_t["amino"].values

    # Combine features and labels for resampling
    train_data = train_t.copy()

    # Separate classes
    balanced_data = []
    for amino_acid in train_data["amino"].unique():
        class_data = train_data[train_data["amino"] == amino_acid]
        n_samples = train_data["amino"].value_counts().max()  # Target class size
        balanced_class_data = resample(
            class_data,
            replace=True,  # Bootstrap resampling
            n_samples=n_samples,  # Match the largest class
            random_state=parameters["seed"]
        )
        balanced_data.append(balanced_class_data)

    # Concatenate all balanced class samples
    balanced_train_data = pd.concat(balanced_data)

    test_drop = test_t.copy().drop(columns=["amino", "protein"])

    nuclei = test_drop.columns.tolist()
    test_t = test_t.dropna(subset=nuclei, how="all")
    train_t = train_t.dropna(subset=nuclei, how="all")

    if algo in ["lda_bootstrap", "filtered_bootstrap"]:
        train_t = balanced_train_data.dropna(subset=nuclei, how="all")

    train_labels = train_t.loc[train_t["protein"] != protein, "amino"].values
    check_labels = test_t.loc[test_t["protein"] == protein, "amino"].values


    test_lab = [None for _ in range(test_t.shape[0])]
    probs = np.ndarray(shape=(len(test_lab), len(AATs_unique)))
    test_nan = np.array(test_t.isnull())
    combs_nan = np.unique(test_nan, axis=0)

    if algo == "sw":
        content["train"] = train_t
        content["test"] = test_t
        content["check_labels"] = check_labels
        content["train_labels"] = train_labels
        content["count_rest"] = AATs_count
        return content

    # Get column positions of GLY and PRO
    if "GLY" in AATs_unique:
        G_idx = AATs_unique.index("GLY")
    if "PRO" in AATs_unique:
        P_idx = AATs_unique.index("PRO")
    # Loop over combinations
    for i in combs_nan:
        idxs_nan = np.all(test_nan == i, axis=1)
        nuclei_meas = list(compress(nuclei, ~i))
        test_missing_comb = test_t[idxs_nan][nuclei_meas]

        # Classify residues with all NaN values
        if np.all(i):
            a = np.empty((idxs_nan.sum(), probs.shape[1]))
            a[:] = np.nan
            probs[idxs_nan, :] = a
            continue

        # Classify residues that could be GLY or PRO
        if {"HB", "CB", "HN"}.isdisjoint(nuclei_meas) and all(
                x in AATs_unique for x in ["GLY", "PRO"]):
            lda = LDA()
            idxs = train_t[nuclei_meas].isnull().any(axis=1)
            train_set_aux = train_t[~idxs][nuclei_meas].copy()
            labels = train_labels[~idxs].copy()

            lda.fit(train_set_aux, labels)
            lda.means_ = 1
            probs[idxs_nan, :] = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))
            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Classify residues that could be GLY
        elif {"HB", "CB"}.isdisjoint(set(nuclei_meas)) and "GLY" in AATs_unique:
            lda = LDA()
            train_set_aux = train_t[train_t["amino"] != "PRO"][nuclei_meas].copy()
            labels = train_labels[train_labels != "PRO"].copy()
            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux[~idxs]
            labels = labels[~idxs]
            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))
            if "PRO" in AATs_unique:
                probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx],
                    np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]
                ]
            else:
                probs[idxs_nan, :] = probs_aux
            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Classify residues that could be PRO
        elif {"HN"}.isdisjoint(set(nuclei_meas)) and "PRO" in AATs_unique:
            lda = LDA()
            train_set_aux = train_t[train_t["amino"] != "GLY"][nuclei_meas].copy()
            labels = train_labels[train_labels != "GLY"].copy()
            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux[~idxs]
            labels = labels[~idxs]

            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))

            if "PRO" not in set(AATs_unique).difference(set(labels)):
                if "GLY" in AATs_unique:
                    probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]
                    ]
                else:
                    probs[idxs_nan, :] = probs_aux
            else:
                if "GLY" in AATs_unique:
                    probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:P_idx - 1],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx - 1:]
                    ]
                else:
                    probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]
                    ]

            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Classify other residues
        else:
            lda = LDA()
            train_set_aux = train_t.query('amino!="PRO" and amino!="GLY"')[nuclei_meas].copy()
            labels = train_labels[
                np.logical_and(train_labels != "PRO", train_labels != "GLY")
            ].copy()
            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux[~idxs]
            labels = labels[~idxs]

            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))
            if all(x in AATs_unique for x in ["GLY", "PRO"]):
                probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, G_idx:P_idx - 1], np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, P_idx - 1:]
                ]
            elif "PRO" in AATs_unique:
                probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx],
                    np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]
                ]
            elif "GLY" in AATs_unique:
                probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx],
                    np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]
                ]
            else:
                probs[idxs_nan, :] = probs_aux
            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

    df_probs = pd.DataFrame(probs, index=test_t.index, columns=AATs_unique)
    df_protein = pd.DataFrame({"real labels": check_labels, "classified as": test_lab, "LDA score": df_probs.max(axis=1).values})
    content["train"] = train_t
    content["test"] = test_t
    content["check_labels"] = check_labels
    content["train_labels"] = train_labels
    content["lda_labels"] = df_protein
    content["lda_probs"] = df_probs
    content["lda_accuracy"] = accuracy(check_labels, test_lab)

    if algo in ["lda", "lda_bootstrap"]:
        df_protein.to_csv(Path(output_dir) / f"labels_{protein}_lda.csv")
        df_probs.to_csv(Path(output_dir) / f"probabilities_{protein}_lda.csv")

    if algo in ["filtered", "filtered_bootstrap"]:
        fltr =  [i
            for i, a in enumerate(AATs_unique)
            if AATs_count.get(a, 0)/len(check_labels) > 0.01
        ]
        protein_filter[protein] = probs[:, fltr].max(axis=1) == 1
        filter_p = pd.DataFrame(protein_filter[protein], columns=[f"{protein}"], )
        content["filter"] = filter_p
        content["count_rest"] = AATs_count - Counter(
            [lab for i, lab in enumerate(test_lab) if protein_filter[protein][i]]
        )
    return content


################################### LDA CLASSIFY ###############################

def lda_classification_classify(
    parameters, excel_protein, scanlist, fasta_path, output_dir
) :
    with open(fasta_path, "r") as f:
        fasta = [line.rstrip("\n") for line in f]
    fasta = "".join(fasta)
    content = {}
    algo = parameters["algo"]
    names = ["H", "HB", "HB1", "HB2", "HB3", "CA", "CB", "C", "CO", "N", "amino", "protein"]

    test_data = pd.read_excel(excel_protein, engine="openpyxl")
    TableB = pd.DataFrame(columns=names)

    for k, l in enumerate(scanlist):
        waiter = True
        sys.stdout.write(f"\nScanning data from BMRB: {k + 1} of {len(scanlist)}")
        sys.stdout.flush()

        while waiter:
            try:
                ent = pynmrstar.Entry.from_database(l)
                waiter = False
            except (ConnectionError, OSError) as _:
                print("\nConnection error on: " + str(l))
                sleep(2)
                waiter = True
        spectral_peaks = ent.get_saveframes_by_category("assigned_chemical_shifts")

        if len(spectral_peaks) > 0:
            TableC = pd.DataFrame(columns=names)
            w = spectral_peaks[0]["_Atom_chem_shift.Seq_ID"]
            x = spectral_peaks[0]["_Atom_chem_shift.Comp_ID"]
            y = spectral_peaks[0]["_Atom_chem_shift.Atom_ID"]
            z = spectral_peaks[0]["_Atom_chem_shift.Val"]
            wp = w[0]
            n = 0

            for i, j in enumerate(w):
                if j != wp:
                    n = n + 1
                    wp = w[i]
                TableC.at[n, "protein"] = l
                TableC.at[n, "amino"] = x[i]
                TableC.at[n, y[i]] = float(z[i])
            TableB = pd.concat([TableB, TableC], ignore_index=True)
    print("\n")
    TableA = pd.DataFrame(
        columns=["amino", "protein", "HN", "N", "CO", "HA", "HB", "CA", "CB"]
    ).astype("float")
    try:
        TableA["HB"] = TableB.loc[:, "HB":"HB3"].astype(float).mean(axis=1).astype(float)
    except KeyError:
        TableA["HB"] = TableB.loc[:, "HB"].astype(float)

    TableA["HN"] = TableB.loc[:, "H"].astype(float)
    TableA["CA"] = TableB.loc[:, "CA"].astype(float)
    TableA["CB"] = TableB.loc[:, "CB"].astype(float)
    TableA["N"] = TableB.loc[:, "N"].astype(float)
    TableA["CO"] = TableB.loc[:, "C"].astype(float)

    try:
        TableA["HA"] = TableB.loc[:, ["HA", "HA2", "HA3"]].astype(float).mean(axis=1)
    except KeyError:
        try:
            TableA["HA"] = TableB.loc[:, "HA"].astype(float)
        except KeyError:
            pass

    TableA["amino"] = TableB["amino"]
    TableA["protein"] = TableB["protein"]

    AAT_dict = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
                "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
                "Y": "TYR", "V": "VAL"}

    # Get Amino Acid Types in test protein from seq file
    AATs_unique = list({AAT_dict[x] for x in set(fasta)})
    AATs_unique.sort()

    check_labels = pd.Series([None for _ in test_data["SSN"].values])

    # Eliminate from training set residues of AATs not present in test protein
    AATs_missing = list(set(AAT_dict.values()) - set(AATs_unique))
    # print(len(AATs_unique))

    AATs_count = Counter([AAT_dict[x] for x in fasta])

    # Test data
    SSN = test_data.iloc[:len(test_data), 0]
    test_set = test_data.drop(["SSN"], axis=1)
    nuclei = test_set.columns.tolist()
    nuclei.remove("amino")

    # Sort training data
    train_data = pd.DataFrame(
        TableA, columns=nuclei+["amino", "protein"]).sort_values(["amino"], ascending=True
    )

    for AAT in AATs_missing: # pylint: disable=unused-variable
        train_data = train_data.drop(train_data.query("amino==@AAT").index)

    # Pareto scaling of train and test data simultaneously
    for_scaling = pd.concat([test_set, train_data], ignore_index=True)
    for k in for_scaling:
        if k in {"amino", "protein"}:
            break
        for_scaling[k] = (
            for_scaling[k] - np.mean(for_scaling[k])) / (np.sqrt(np.std(for_scaling[k]))
        )


    test_t = for_scaling.iloc[:len(test_data), :]
    train_t = for_scaling.iloc[len(test_data):, :]

    # Balancing data using bootstrap resampling
    train_labels = train_t["amino"].values

    # Combine features and labels for resampling
    train_data = train_t.copy()

    # Separate classes
    balanced_data = []
    for amino_acid in train_data["amino"].unique():
        class_data = train_data[train_data["amino"] == amino_acid]
        n_samples = train_data["amino"].value_counts().max()  # Target class size
        balanced_class_data = resample(
            class_data,
            replace=True,  # Bootstrap resampling
            n_samples=n_samples,  # Match the largest class
            random_state=parameters["seed"]
        )
        balanced_data.append(balanced_class_data)

    # Concatenate all balanced class samples
    balanced_train_data = pd.concat(balanced_data)

    test_drop = test_t.copy().drop(columns=["amino", "protein"])

    nuclei = test_drop.columns.tolist()
    test_t = test_t.dropna(subset=nuclei, how="all")
    train_t = train_t.dropna(subset=nuclei, how="all")

    if algo in ["lda_bootstrap", "filtered_bootstrap"]:
        train_t = balanced_train_data.dropna(subset=nuclei, how="all")

    train_labels = train_t["amino"].values

    test_lab = [None for _ in range(test_t.shape[0])]
    probs = np.ndarray(shape=(len(test_lab), len(AATs_unique)))
    test_nan = np.array(test_t.isnull())
    combs_nan = np.unique(test_nan, axis=0)

    if algo == "sw":
        content["train"] = train_t
        content["test"] = test_t
        content["check_labels"] = check_labels
        content["train_labels"] = train_labels
        content["count_rest"] = AATs_count
        return content

    # Get column positions of GLY and PRO
    if "GLY" in AATs_unique:
        G_idx = AATs_unique.index("GLY")
    if "PRO" in AATs_unique:
        P_idx = AATs_unique.index("PRO")
    # Loop over combinations
    for i in combs_nan:
        idxs_nan = np.all(test_nan == i, axis=1)
        nuclei_meas = list(compress(nuclei, ~i))
        test_missing_comb = test_t[idxs_nan][nuclei_meas]

        # Classify residues with all NaN values
        if np.all(i):
            a = np.empty((idxs_nan.sum(), probs.shape[1]))
            a[:] = np.nan
            probs[idxs_nan, :] = a
            continue

        # Classify residues that could be GLY or PRO
        if {"HB", "CB", "HN"}.isdisjoint(nuclei_meas) and all(
                x in AATs_unique for x in ["GLY", "PRO"]):
            lda = LDA()
            idxs = train_t[nuclei_meas].isnull().any(axis=1)
            train_set_aux = train_t[~idxs][nuclei_meas].copy()
            labels = train_labels[~idxs].copy()

            lda.fit(train_set_aux, labels)
            lda.means_ = 1
            probs[idxs_nan, :] = lda.predict_proba(test_missing_comb)
            test = test_missing_comb.values
            test_labels = list(lda.predict(test_missing_comb))
            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Classify residues that could be GLY
        elif {"HB", "CB"}.isdisjoint(set(nuclei_meas)) and "GLY" in AATs_unique:
            lda = LDA()
            train_set_aux = train_t[train_t["amino"] != "PRO"][nuclei_meas].copy()
            labels = train_labels[train_labels != "PRO"].copy()
            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux[~idxs]
            labels = labels[~idxs]
            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))
            if "PRO" in AATs_unique:
                probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx],
                    np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]
                ]
            else:
                probs[idxs_nan, :] = probs_aux
            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Classify residues that could be PRO
        elif {"HN"}.isdisjoint(set(nuclei_meas)) and "PRO" in AATs_unique:
            lda = LDA()
            train_set_aux = train_t[train_t["amino"] != "GLY"][nuclei_meas].copy()
            labels = train_labels[train_labels != "GLY"].copy()
            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux[~idxs]
            labels = labels[~idxs]

            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))

            if "PRO" not in set(AATs_unique).difference(set(labels)):
                if "GLY" in AATs_unique:
                    probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]
                    ]
                else:
                    probs[idxs_nan, :] = probs_aux
            else:
                if "GLY" in AATs_unique:
                    probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:P_idx - 1],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx - 1:]
                    ]
                else:
                    probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx],
                        np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]
                    ]

            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

        # Classify other residues
        else:
            lda = LDA()
            train_set_aux = train_t.query('amino!="PRO" or amino!="GLY"')[nuclei_meas].copy()
            labels = train_labels[
                np.logical_or(train_labels != "PRO", train_labels != "GLY")
            ].copy()
            idxs = train_set_aux.isnull().any(axis=1)
            train_set_aux = train_set_aux[~idxs]
            labels = labels[~idxs]

            lda.fit(train_set_aux, labels)
            probs_aux = lda.predict_proba(test_missing_comb)
            test_labels = list(lda.predict(test_missing_comb))
            if all(x in AATs_unique for x in ["GLY", "PRO"]):
                probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, G_idx:P_idx - 1], np.zeros((idxs_nan.sum(), 1)),
                    probs_aux[:, P_idx - 1:]
                ]
            elif "PRO" in AATs_unique:
                probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx], \
                    np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]]
            elif "GLY" in AATs_unique:
                probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], \
                    np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]]
            else:
                probs[idxs_nan, :] = probs_aux
            for idx, val in enumerate(idxs_nan):
                if val:
                    test_lab[idx] = test_labels.pop(0)

    df_probs = pd.DataFrame(probs, index=test_t.index, columns=AATs_unique)
    df_protein = pd.DataFrame({"real labels": check_labels, "classified as": test_lab, "LDA score": df_probs.max(axis=1).values})
    content["train"] = train_t
    content["test"] = test_t
    content["train_labels"] = train_labels
    content["lda_labels"] = df_protein
    content["lda_probs"] = df_probs

    if algo == "lda":
        df_protein.to_csv(Path(output_dir) / "labels_lda.csv")
        df_probs.to_csv(Path(output_dir) / "probabilities_lda.csv")

    if algo == "lda_bootstrap":
        df_protein.to_csv(Path(output_dir) / "labels_lda_bootstrap.csv")
        df_probs.to_csv(Path(output_dir) / "probabilities_lda_bootstrap.csv")

    if algo in ["filtered", "filtered_bootstrap"]:
        fltr = [i
            for i, a in enumerate(AATs_unique)
            if AATs_count.get(a, 0) / len(check_labels) > 0.01
        ]
        protein_filter = probs[:, fltr].max(axis=1) == 1
        filter_p = pd.DataFrame(protein_filter)
        content["filter"] = filter_p
        cnt_rest = Counter([lab for i, lab in enumerate(test_lab) if protein_filter[i]])
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
    comb, parameters, params_comb
):
    n_projections = parameters["n_projections"]
    seed = parameters["seed"]
    k = parameters["k"]
    device = parameters["device"]
    with_bins = parameters["with_bins"]
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

    if vote_type == "NORMAL":
        test_labels, total_transport_classes = classify_sw(
            x1_torch, x2_torch, train_comb_labels, classes=ALL_POSSIBLE,
            n_projections=n_projections, seed=gen, device=device, scale=scale, k=k, kappa1=kappa1,
            kappa2=kappa2, with_bins=with_bins
        )
    elif vote_type == "SPECIAL":
        test_labels, total_transport_classes = classify_sw_spec(
            x1_torch, x2_torch, train_comb_labels, check_comb_labels, classes = ALL_POSSIBLE,
            dimPRO=dimPRO, dimGLY=dimGLY, n_projections=n_projections, n_specPRO=n_specPRO,
            n_specGLY=n_specGLY, parameters_amino=params_comb, seed=gen, device=device,
            scale=scale, k=k, kappa1=kappa1, kappa2=kappa2, with_bins=with_bins
        )
    else:
        assert False, "Only NORMAL or SPECIAL"

    li = 0
    for idx, val in enumerate(idxs_nan):
        if val and test_labels:
            test_lab[idx] = test_labels.pop(0)
            prob_matrix[:, idx] = total_transport_classes[:, li]
            li +=1
    return (test_lab, prob_matrix)

def validate(i, train, test, parameters, parameters_amino, param_str, output_dir):
    train_labels, check_labels, test_drop, nuclei = prepare_labels(train, test)
    nuclei = ["HN", "N", "CO", "CA", "CB", "HA", "HB"]
    useful_structs = useful_structures(test_drop)
    test_lab, test_nan, combs_nan, prob_matrix = useful_structs

    for k, comb in enumerate(combs_nan):
        print(f"{datetime.now().time()} Task {i} in comb {k+1} of {len(combs_nan)}.", flush=True)
        result = classify_in_combination(train, train_labels, test, check_labels, nuclei, test_lab,\
                             test_nan, prob_matrix, comb, parameters, parameters_amino)
        if result is None:
            continue

        test_lab, prob_matrix = result


    df_prob = pd.DataFrame(
        normalize_matrix_col(prob_matrix), index = ALL_POSSIBLE, columns=check_labels
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
def combine_methods(output_dir, content_lda, content_sw, param_str):
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
    AATs_count = Counter(lda_labels["real labels"].values)
    best_method  = None
    methods = []
    # Adjust SW label classification based on LDA and SW scores (probabilities)
    for i in range(len(sw_labels.index)):
        best_label = sw_labels["classified as"].iloc[i]
        best_method = 'SW'
        if prob_lda.max(axis=1).iloc[i] >= 0.55:
            if AATs_count[prob_lda.idxmax(axis=1).iloc[i]]/len(lda_labels.index) >= 0.01:
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
            prob_values = prob_sw.iloc[:, i].to_numpy()
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
