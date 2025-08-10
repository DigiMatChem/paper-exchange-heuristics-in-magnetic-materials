"""
Functions to compute coordination-based parallelity and antiparallelity scores
both as scalars with different bond / site weighting and as site-resolved target.
"""
import numpy as np
from pymatgen.core.structure import Structure

from utils_kga.general import get_angle_between
from utils_kga.coordination_features import CoordinationFeatures


def get_p_ap_scores(structure: Structure,
                    coordination_features: CoordinationFeatures,
                    spin_angle_tol: float = 10.0,
                    weighting: str = "include_ligand_multiplicities"):
    """
    Computes p and ap score weighted by bond multiplicity and number of
    connecting ligands.
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param spin_angle_tol: maximum absolute angle in deg to count two magnetic vectors
        as parallel / antiparallel
    :param weighting: weighting strategy for score computation.
    :return: tuple (p, ap) | None
    """
    # Define weighting factor and check parameter input
    weighting_factors = {"include_ligand_multiplicities": lambda x: x[1],
                         "exclude_ligand_multiplicities": lambda x: 1}
    if weighting not in weighting_factors:
        raise KeyError(f"Weighting parameter must be in {list(weighting_factors.keys())}")

    sitewise_spin_angles = get_sitewise_spin_angles(structure=structure, coordination_features=coordination_features)
    spin_angles = []
    for value in sitewise_spin_angles.values():
        spin_angles.extend(value)

    if spin_angles:
        # Compute p and ap scores for given angle tolerance and weighting
        p_spins = sum([weighting_factors[weighting](sp) for sp in spin_angles if sp[0] <= spin_angle_tol])
        ap_spins = sum([weighting_factors[weighting](sp) for sp in spin_angles if sp[0] >= (180.0 - spin_angle_tol)])
        all_spins = sum([weighting_factors[weighting](sp) for sp in spin_angles])

        p_score = p_spins / all_spins
        ap_score = ap_spins / all_spins

        return p_score, ap_score

    elif any(np.concatenate([site.properties["magmom"].get_moment() for site in structure.sites])):
        return -1, -1


def get_sitewise_p_ap_scores(structure: Structure,
                             coordination_features: CoordinationFeatures,
                             spin_angle_tol: float = 10.0,
                             weighting: str = "include_ligand_multiplicities"):
    """
    Computes p and ap score weighted by bond multiplicity and number of
    connecting ligands.
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param spin_angle_tol: maximum absolute angle in deg to count two magnetic vectors
        as parallel / antiparallel
    :param weighting: weighting strategy for score computation.
    :return: structure decorated with sitewise p, ap score (as site properties)
    """
    # Define weighting factor and check parameter input
    weighting_factors = {"include_ligand_multiplicities": lambda x: x[1],
                         "exclude_ligand_multiplicities": lambda x: 1}
    if weighting not in weighting_factors:
        raise KeyError(f"Weighting parameter must be in {list(weighting_factors.keys())}")

    structure_copy = structure.copy()

    sitewise_spin_angles = get_sitewise_spin_angles(structure=structure_copy, coordination_features=coordination_features)

    for cation_id, spin_angles in sitewise_spin_angles.items():
        if spin_angles:
            p_spins = sum([weighting_factors[weighting](sp) for sp in spin_angles if sp[0] <= spin_angle_tol])
            ap_spins = sum(
                [weighting_factors[weighting](sp) for sp in spin_angles if sp[0] >= (180.0 - spin_angle_tol)])
            all_spins = sum([weighting_factors[weighting](sp) for sp in spin_angles])

            p_score = p_spins / all_spins
            ap_score = ap_spins / all_spins

            structure_copy.sites[cation_id].properties["p_score"] = p_score
            structure_copy.sites[cation_id].properties["ap_score"] = ap_score

        # Handle isolated magnetic sites (denote as -1)
        elif any(structure_copy.sites[cation_id].properties["magmom"].get_moment()):
            structure_copy.sites[cation_id].properties["p_score"] = -1
            structure_copy.sites[cation_id].properties["ap_score"] = -1

    return structure_copy


def get_sitewise_spin_angles(structure: Structure,
                             coordination_features: CoordinationFeatures):
    """
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :return: dict of {cation_id: list of [spin_angle, len(n["ligand_indices"])]
    """
    # Sanity check: quickly assert matching Structure and CoordinationFeatures object
    for site_idx, site in enumerate(structure.sites):
        for coord in range(3):
            assert round(site.coords[coord], 4) == round(coordination_features.sites.coordinates[site_idx][coord], 4), \
                f"Different sites in Structure {structure.formula} and CoordinationFeature object!"

    cation_ids = [ce["site"] for ce in coordination_features.ces]
    sitewise_spin_angles = {cation: [] for cation in cation_ids}

    for n in coordination_features.ce_neighbors:

        # Only include connections between sites with non-zero magnetic moments
        v1 = structure.sites[n["site"]].properties["magmom"].get_moment()
        v2 = structure.sites[n["site_to"]].properties["magmom"].get_moment()
        if np.any(v1) and np.any(v2):
            spin_angle = get_angle_between(v1, v2)
            sitewise_spin_angles[n["site"]].append([spin_angle, len(n["ligand_indices"])])

    return sitewise_spin_angles
