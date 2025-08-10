from collections import Counter
import numpy as np
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure, Element
from typing import Callable, Tuple

from utils_kga.featurization.sitewise_coordinationnet_features import get_node_features as get_node_infos
from utils_kga.featurization.sitewise_coordinationnet_features import (
    get_neighbor_features as get_sitewise_neighbor_features)
from utils_kga.featurization.sitewise_coordinationnet_features import (d_f_cation_neighbor_filter,
                                                                       magnetic_cation_neighbor_filter)
from utils_kga.coordination_features import CoordinationFeatures


stats_map = {"mean": np.mean, "median": np.median, "std": np.std, "min": np.min, "max": np.max,
             "most_frequent": lambda ls: get_most_frequent_item(ls)}


def get_structurewise_coordinationnet_features(structure: Structure,
                                               coordination_features: CoordinationFeatures,
                                               site_filters: Tuple[Callable, ...],
                                               stats: Tuple[str, ...],
                                               magmom_info: tuple) -> dict:
    """
    Compute structurewise coordinatinonet-derived features by applying statistical measures to arrays of
    node-/edge-infos.
    :param structure: pymatgen Structure object without magmom information
    :param coordination_features: CoordinationFeatures object
    :param site_filters: selection functions for magnetic site features
    :param stats: string representations of what statistic measures to compute
    :param magmom_info: tuple of usually magnetic elements / species, e.g. as in pymatgen's default_magmoms
    :return: dict of structurewise coordinationnet features
    """
    all_features = {}
    sga = SpacegroupAnalyzer(structure=structure, symprec=1e-3)
    symm_structure = sga.get_symmetrized_structure()

    for site_filter in site_filters:
        suffix = [p[1] for p in cation_filter_suffix_map if p[0] == site_filter][0]
        all_scalar_site_infos, all_ligand_distances = [], []
        for eq_group in symm_structure.equivalent_indices:  # Get unique sites and their multiplicity to save time
            site_idx = eq_group[0]
            multiplicity = len(eq_group)
            if site_filter(site_idx=site_idx, coordination_features=coordination_features, magmom_info=magmom_info):
                scalar_site_info = get_node_infos(site_idx=site_idx, coordination_features=coordination_features)
                edge_info, ligand_distances = get_edge_infos(
                    site_idx=site_idx, coordination_features=coordination_features)
                scalar_site_info.update(edge_info)

                all_scalar_site_infos.extend([scalar_site_info for _ in range(multiplicity)])
                all_ligand_distances.extend(ligand_distances * multiplicity)

        # If no sites guessed magnetic (due to Re/Pu compounds in dataset), replace with d f filter features
        if not all_scalar_site_infos:
            all_features.update({k.replace("_dfcat", "_magcat"): v for k, v in all_features.items()})
            continue
        all_scalar_site_infos = {k: [dic[k] for dic in all_scalar_site_infos] for k in all_scalar_site_infos[0]}
        all_scalar_site_infos.update({"cn_ligand_distances": all_ligand_distances})
        feats = get_features_from_dict_of_lists(d=all_scalar_site_infos, stats=stats)
        feats.update(get_neighbor_features(coordination_features=coordination_features, site_filter=site_filter,
                                           stats=stats, magmom_info=magmom_info))
        all_features.update({k.removeprefix("site_") + suffix: v for k, v in feats.items()})
    return all_features


def get_edge_infos(site_idx: int,
                   coordination_features: CoordinationFeatures) -> (dict, list):
    """
    Extract edge-related infos for a selected site in a structure. Not final featurization, later
    computation of statistical measures of list of infos in get_structurewise_coordinationnet_features.
    :param site_idx: index of site in the pymatgen Structure object
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :return: dict of the form {info_key: info_values}, ligand distances
    """
    csm = [i["csms"][0] for i in coordination_features.ces if i["site"] == site_idx][0]
    ligands = [(Element(coordination_features.sites.elements[d["site_to"]]).Z, d["distance"]) for d
               in coordination_features.distances if d["site"] == site_idx]
    scalar_site_info = {"cn_csm": csm if csm else 0.0,
                        "cn_mean_Z_ligands": np.mean(np.array([li[0] for li in ligands])),}
    return scalar_site_info, [li[1] for li in ligands]


def get_neighbor_features(coordination_features: CoordinationFeatures, site_filter: Callable,
                          stats: Tuple[str, ...], magmom_info: tuple = tuple()) -> dict:
    """
    Get information on distances, bond angles and connectivities of sites guessed magnetic. Also adapt
    sitewise featurization function for different "weighting" of neighbor features.
    :param coordination_features: a pycoordinationnet CoordinationFeatures object
    :param site_filter: function for filtering sites to featurize
    :param stats: string representations of what statistic measures to compute
    :param magmom_info: tuple of usually magnetic elements / species, e.g. as in pymatgen's default_magmoms
    :return: dict of features on connectivities, neighbor distances and bond angles
    """
    connectivities = []
    neighbor_infos = {"cn_bond_angles": [], "cn_neighbor_distances": []}
    has_mag_connections = True
    for n in coordination_features.ce_neighbors:
        if (site_filter(site_idx=n["site"], coordination_features=coordination_features, magmom_info=magmom_info)
                and site_filter(
                    site_idx=n["site_to"], coordination_features=coordination_features, magmom_info=magmom_info)):
            neighbor_infos["cn_bond_angles"].extend(n["angles"])
            neighbor_infos["cn_neighbor_distances"].append(n["distance"])
            connectivities.append(n["connectivity"])

    if not connectivities:  # Isolated sites
        has_mag_connections = False
        neighbor_infos = {"cn_bond_angles": [0.0], "cn_neighbor_distances": [0.0]}
        connectivities = ["none"]

    feat_dict = {
        "cn_corner_perc": connectivities.count("corner") / len(connectivities),
        "cn_edge_perc": connectivities.count("edge") / len(connectivities),
        "cn_face_perc": connectivities.count("face") / len(connectivities)}
    feat_dict.update(get_features_from_dict_of_lists(d=neighbor_infos, stats=stats))
    feat_dict.update({"cn_has_mag_connections": int(has_mag_connections)})

    # Create features from sitewise handling of edge information
    all_nb_site_infos = []
    for site_idx in coordination_features.sites.sites:
        if site_filter(site_idx=site_idx, coordination_features=coordination_features, magmom_info=magmom_info):
            all_nb_site_infos.append(get_sitewise_neighbor_features(site_idx=site_idx,
                                                                    coordination_features=coordination_features,
                                                                    site_filter_neighbors=(
                                                                        cation_filter_structure_site_map[site_filter],),
                                                                    magmom_info=magmom_info))
    all_nb_site_keys = set().union(*(dic.keys() for dic in all_nb_site_infos))
    all_nb_site_infos = {k: [dic[k] for dic in all_nb_site_infos if k in dic] for k in all_nb_site_keys}
    feats = get_features_from_dict_of_lists(d=all_nb_site_infos, stats=stats)
    feat_dict.update(feats)

    return feat_dict


def get_features_from_dict_of_lists(d: dict, stats: Tuple[str, ...]) -> dict:
    """
    :param d: dict of infos in lists
    :param stats: string representations of what statistic measures to compute
    :return: dict of features
    """
    feat_dict = {}
    for k, ls in d.items():
        ls = [x for x in ls if x is not None]
        if ls:
            feat_dict.update({k + "_" + stat: stats_map[stat](np.array(ls)) for stat in stats
                              if stat != "most_frequent"})
            if "most_frequent" in stats:
                rounded_ls = [round(item, 0) for item in ls] \
                    if k.startswith("cn_bond_angles") else [round(item, 2) for item in ls]
                feat_dict.update({k + "_most_frequent": get_most_frequent_item(rounded_ls=rounded_ls)})
    return feat_dict


def get_most_frequent_item(rounded_ls: list) -> float:
    d = Counter(rounded_ls)
    return max(d, key=d.get)


# As opposed to sitewise ML, apply same filters to site and neighbors
def d_f_cation_filter(site_idx: int, coordination_features: CoordinationFeatures, magmom_info: tuple) -> bool:
    return (Element(coordination_features.sites.elements[site_idx]).block in ["d", "f"]
            and coordination_features.sites.ions[site_idx] == "cation")


def magnetic_cation_filter(site_idx: int, coordination_features: CoordinationFeatures, magmom_info: tuple) -> bool:
    return (coordination_features.sites.elements[site_idx] in magmom_info
            and coordination_features.sites.ions[site_idx] == "cation")


cation_filter_suffix_map = ((d_f_cation_filter, "_dfcat"),
                            (magnetic_cation_filter, "_magcat"))

cation_filter_structure_site_map = {d_f_cation_filter: d_f_cation_neighbor_filter,
                                    magnetic_cation_filter: magnetic_cation_neighbor_filter}
