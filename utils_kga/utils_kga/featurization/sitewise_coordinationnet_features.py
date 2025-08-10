"""
Utilities for site-wise features based on pymatgen Structure and
pycoordinationnet CoordinationFeatures objects.
Returns sitewise dict of features for all sites where site_filter condition applies.
"""
from collections import Counter
import numpy as np
from pymatgen.core import Structure, Element, Species
from typing import Callable, Tuple

from utils_kga.coordination_features import CoordinationFeatures


CN_NOTATION_MAP = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII", "8": "VIII", "9": "IX"}
BLOCK_MAP = {"s": 0, "p": 1, "d": 2, "f": 3}


def get_sitewise_coordinationnet_features(structure: Structure,
                                          coordination_features: CoordinationFeatures,
                                          site_filter_neighbors: Tuple[Callable, ...],
                                          ineq_site_ids: list,
                                          magmom_info: tuple) -> dict:
    """
    Extract CoordinationFeatures-based features for selected sites in a structure.
    :param structure: a pymatgen Structure object without magnetic information
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param site_filter_neighbors: neighbor selection functions for neighbor features
    :param ineq_site_ids: indices of crystallographically unique sites of the non-magnetic phase
    :param magmom_info: tuple of usually magnetic elements / species, e.g. as in pymatgen's default_magmoms
    :return: dict of the form {site_idx: {feature_key: feature_values}}
    """
    feat_dict = {}
    for site_idx, site in enumerate(structure.sites):
        # Collect node information about all cationic d- and f-block sites that are cryst. unique
        if unique_d_f_cation_filter(site_idx=site_idx, coordination_features=coordination_features,
                                    ineq_site_ids=ineq_site_ids):
            site_dict = get_node_features(site_idx=site_idx, coordination_features=coordination_features)
            site_dict.update(get_edge_features(site_idx=site_idx, coordination_features=coordination_features))
            site_dict.update(get_neighbor_features(site_idx=site_idx,
                                                   coordination_features=coordination_features,
                                                   site_filter_neighbors=site_filter_neighbors,
                                                   magmom_info=magmom_info))

            feat_dict.update({site_idx: site_dict})
    return feat_dict


def get_node_features(site_idx: int,
                      coordination_features: CoordinationFeatures) -> dict:
    """
    Extract node-related (including only site) features for a selected site in a structure.
    :param site_idx: index of site to be featurized in the pymatgen Structure object
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :return: dict of the form {feature_key: feature_values}
    """
    element = Element(coordination_features.sites.elements[site_idx])
    species = Species(symbol=coordination_features.sites.elements[site_idx],
                      oxidation_state=coordination_features.sites.oxidations[site_idx])
    return {
        "site_cn_element": element.Z,
        "site_cn_oxidation": coordination_features.sites.oxidations[site_idx],
        "site_cn_X": element.X,
        "site_cn_atomic_radius": element.atomic_radius,
        "site_cn_ionization_energy": element.ionization_energy,
        "site_cn_av_cationic_radius": element.average_cationic_radius,
        "site_cn_row": element.row,
        "site_cn_group": element.group,
        "site_cn_block": BLOCK_MAP[element.block],
        "site_cn_is_transition_metal": int(element.is_transition_metal),
        "site_cn_species_ionic_radius": species.ionic_radius if species.ionic_radius else 0.0,
            }


def get_edge_features(site_idx: int,
                      coordination_features: CoordinationFeatures) -> dict:
    """
    Extract edge-related features for a selected site in a structure.
    :param site_idx: index of site to be featurized in the pymatgen Structure object
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :return: dict of the form {feature_key: feature_values}
    """
    csm = [i["csms"][0] for i in coordination_features.ces if i["site"] == site_idx][0]
    ce = [i["ce_symbols"][0] for i in coordination_features.ces if i["site"] == site_idx][0]
    site_dict = {"site_cn_csm": csm if csm else 0}

    ligands = [(Element(coordination_features.sites.elements[d["site_to"]]).Z, d["distance"]) for d
               in coordination_features.distances if d["site"] == site_idx]
    ligand_distances = np.array([li[1] for li in ligands])

    site_dict.update({
        "site_cn_n_ligands": len(ligands),
        "site_cn_mean_Z_ligands": np.mean(np.array([li[0] for li in ligands])),
        "site_cn_mean_lig_distance": np.mean(ligand_distances),
        "site_cn_min_lig_distance": np.min(ligand_distances),
        "site_cn_max_lig_distance": np.max(ligand_distances),
        "site_cn_median_lig_distance": np.median(ligand_distances),
        "site_cn_std_lig_distance": np.std(ligand_distances),
    })

    species = Species(symbol=coordination_features.sites.elements[site_idx],
                      oxidation_state=coordination_features.sites.oxidations[site_idx])
    # If tet/oct, get crystal field spin by pymatgen
    if ce == "O:6":
        try:
            site_dict["site_cn_crystal_field_spin"] = species.get_crystal_field_spin(coordination="oct")
        except AttributeError:
            site_dict["site_cn_crystal_field_spin"] = -1
    elif ce == "T:4":
        try:
            site_dict["site_cn_crystal_field_spin"] = species.get_crystal_field_spin(coordination="tet")
        except AttributeError:
            site_dict["site_cn_crystal_field_spin"] = -1
    else:
        site_dict["site_cn_crystal_field_spin"] = -1

    return site_dict


def get_neighbor_features(site_idx: int,
                          coordination_features: CoordinationFeatures,
                          site_filter_neighbors: Tuple[Callable, ...],
                          magmom_info: tuple = tuple()
                          ) -> dict:
    """
    Extract neighbor-related features for a selected site in a structure.
    :param site_idx: index of site to be featurized in the pymatgen Structure object
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param site_filter_neighbors: neighbor selection functions for neighbor-derived features like bond angles
    :param magmom_info: tuple of usually magnetic elements / species, e.g. as in pymatgen's default_magmoms
    :return: dict of the form {feature_key: feature_values}
    """
    all_feat_dict = {}

    for neighbor_filter in site_filter_neighbors:
        has_mag_neighbors = True
        bond_angles, connectivities, neighbor_distances, neighbor_features = [], [], [], []
        suffix = [p[1] for p in neighbor_filter_suffix_map if p[0] == neighbor_filter][0]  # bit hacky
        for n_idx, n in enumerate(coordination_features.ce_neighbors):
            if n["site"] == site_idx:
                if neighbor_filter(site_idx=n["site_to"], coordination_features=coordination_features,
                                   magmom_info=magmom_info):
                    bond_angles.extend(n["angles"])
                    connectivities.append(n["connectivity"])
                    neighbor_distances.append(n["distance"])
                    # neighbor_features.append(get_node_features(site_idx=n["site_to"],
                    #                                           coordination_features=coordination_features))

        if not bond_angles:  # isolated_sites
            has_mag_neighbors = False
            bond_angles, neighbor_distances = [0.0], [0,0]
            connectivities = ["none"]

        bond_angles = np.array(bond_angles)
        neighbor_distances = np.array(neighbor_distances)
        angle_occus = Counter([round(ba, 0) for ba in bond_angles])
        most_common_angle = np.array([k for k, v in angle_occus.items() if v == max(angle_occus.values())])

        feat_dict = {
            "site_cn_min_bond_angle": np.min(bond_angles),
            "site_cn_mean_bond_angle": np.mean(bond_angles),
            "site_cn_median_bond_angle": np.median(bond_angles),
            "site_cn_max_bond_angle": np.max(bond_angles),
            "site_cn_std_bond_angle": np.std(bond_angles),
            "site_cn_most_frequent_rounded_bond_angle": np.mean(most_common_angle),

            "site_cn_min_neighbor_distance": np.min(neighbor_distances),
            "site_cn_mean_neighbor_distance": np.mean(neighbor_distances),
            "site_cn_median_neighbor_distance": np.median(neighbor_distances),
            "site_cn_max_neighbor_distance": np.max(neighbor_distances),
            "site_cn_std_neighbor_distance": np.std(neighbor_distances),

            "site_cn_corner_perc": connectivities.count("corner") / len(connectivities),
            "site_cn_edge_perc": connectivities.count("edge") / len(connectivities),
            "site_cn_face_perc": connectivities.count("face") / len(connectivities),

            "site_cn_has_mag_neighbors": int(has_mag_neighbors)
        }

        all_feat_dict.update({k + suffix: v for k, v in feat_dict.items()})

    return all_feat_dict


# For site selection
def unique_d_f_cation_filter(site_idx: int, coordination_features: CoordinationFeatures,
                             ineq_site_ids: list) -> bool:
    """Filter used for site (not neighbor!) featurization: The respective site is crystallographically unique,
    is a d or f block element and recognized cationic by pycoordinationnet."""
    return (site_idx in ineq_site_ids
            and Element(coordination_features.sites.elements[site_idx]).block in ["d", "f"]
            and coordination_features.sites.ions[site_idx] == "cation")


# For neighbor selection
def d_f_cation_neighbor_filter(site_idx: int, coordination_features: CoordinationFeatures,
                               magmom_info: tuple) -> bool:
    return (Element(coordination_features.sites.elements[site_idx]).block in ["d", "f"]
            and coordination_features.sites.ions[site_idx] == "cation")


def magnetic_neighbor_filter(site_idx: int, coordination_features: CoordinationFeatures,
                             magmom_info: tuple) -> bool:
    return coordination_features.sites.elements[site_idx] in magmom_info


def magnetic_cation_neighbor_filter(site_idx: int, coordination_features: CoordinationFeatures,
                                    magmom_info: tuple) -> bool:
    return (coordination_features.sites.elements[site_idx] in magmom_info
            and coordination_features.sites.ions[site_idx] == "cation")


neighbor_filter_suffix_map = ((d_f_cation_neighbor_filter, "_dfcat_nb"),
                              (magnetic_neighbor_filter, "_mag_nb"),
                              (magnetic_cation_neighbor_filter, "_magcat_nb"))
