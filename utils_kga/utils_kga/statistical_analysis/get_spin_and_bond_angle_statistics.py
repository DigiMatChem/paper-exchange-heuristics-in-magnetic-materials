"""Utilites for analyzing spin- and bond angle-related trends in crystallographically unique MAGNDATA dataset."""
from collections import Counter
import numpy as np
import pandas as pd
from pymatgen.core import Structure

from utils_kga.general import get_angle_between
from utils_kga.data_retrieval_and_preprocessing.get_p_ap_scores import get_sitewise_p_ap_scores
from utils_kga.coordination_features import CoordinationFeatures


# Globals for custom binning of p, ap and collinearity binning for including isolated magnetic sites
P_AP_INTERVALS = pd.interval_range(start=-1e-7, end=1.0, periods=10)
P_AP_INTERVAL_STRINGS = ["[0.0-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)", "[0.4-0.5)",
                         "[0.5-0.6)", "[0.6-0.7)", "[0.7-0.8)", "[0.8-0.9)", "[0.9-1.0]"]
P_AP_INTERVAL_MAP = dict(zip(P_AP_INTERVALS, P_AP_INTERVAL_STRINGS))
P_AP_INTERVAL_MAP.update({np.NaN: "isolated"})


def get_magnetic_edge_information(structure: Structure,
                                  coordination_features: CoordinationFeatures,
                                  only_include_pure_oxygen_edges: bool = False,
                                  spin_angle_tols: tuple[float, ...] = (0.0, 10.0, 20.0),
                                  ) -> pd.DataFrame:
    """
    Extract edge information (spin and bond angles, ligand types etc. ) of connected magnetic sites from magnetic
    structure and its respective CoordinationFeatures object (needs to include all ligand images!).
    Intended for statistical analysis and as benchmarking of guessing methods, NOT as actual featurization!
    (As magnetic sites are not guessed).
    Note: SimplestChemEnv strategy for constructing CoordinationFeature object is assumed.
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param only_include_pure_oxygen_edges: whether to only add edges to structure df if all ligand elements
    of the respective edge are oxygen.
    :param spin_angle_tols: maximum absolute angles in deg to count two magnetic vectors
    as parallel / antiparallel.
    :return: pandas.DataFrame, each row represents one edge, with infos on relative spin and bond angles,
    node infos like ces etc. for graph representation of structure filtered for edges where both nodes are magnetic.
    """
    # Avoid unnecessary iterations
    if only_include_pure_oxygen_edges and not structure.composition.__contains__("O"):
        return pd.DataFrame.from_records([])

    oxygen_bool_function = {True: lambda element_list: set(element_list) == set("O"),
                            False: lambda _: True}

    decorated_structures = compute_p_ap_decorated_structures(structure=structure,
                                                             coordination_features=coordination_features,
                                                             spin_angle_tols=spin_angle_tols)

    # Store relevant information in dataframe
    row_lists = []
    for n in coordination_features.ce_neighbors:
        ligand_elements = [coordination_features.sites.elements[i] for i in n["ligand_indices"]]
        if oxygen_bool_function[only_include_pure_oxygen_edges](ligand_elements):

            # Only include connections between site with non-zero magnetic moments
            v1 = structure.sites[n["site"]].properties["magmom"].get_moment()
            v2 = structure.sites[n["site_to"]].properties["magmom"].get_moment()

            if np.any(v1) and np.any(v2):
                interaction_dict = {"site": n["site"],
                                    "site_element": coordination_features.sites.elements[n["site"]],
                                    "site_oxidation": coordination_features.sites.oxidations[n["site"]],
                                    "site_ce": [i["ce_symbols"][0] for i in coordination_features.ces
                                              if i["site"] == n["site"]][0],
                                    "site_csm": [i["csms"][0] for i in coordination_features.ces
                                               if i["site"] == n["site"]][0],
                                    "site_to": n["site_to"],
                                    "site_to_element": coordination_features.sites.elements[n["site_to"]],
                                    "site_to_oxidation": coordination_features.sites.oxidations[n["site_to"]],
                                    "site_to_ce": [i["ce_symbols"][0] for i in coordination_features.ces
                                                 if i["site"] == n["site_to"]][0],
                                    "site_to_csm": [i["csms"][0] for i in coordination_features.ces
                                                  if i["site"] == n["site_to"]][0],
                                    "spin_angle": get_angle_between(v1, v2),
                                    "connectivity": n["connectivity"],
                                    "ligand_elements": ligand_elements,
                                    "ligand_indices": n["ligand_indices"],
                                    "mag_ligand_mag_angle": n["angles"],
                                    "distance": n["distance"]
                                    }

                for spin_angle_tol in spin_angle_tols:
                    for weighting in ["include_ligand_multiplicities", "exclude_ligand_multiplicities"]:
                        interaction_dict.update({
                            f"site_p_{spin_angle_tol}_{weighting}":
                                decorated_structures[spin_angle_tol][weighting].site_properties["p_score"][n["site"]],
                            f"site_ap_{spin_angle_tol}_{weighting}":
                                decorated_structures[spin_angle_tol][weighting].site_properties["ap_score"][n["site"]],
                            f"site_to_p_{spin_angle_tol}_{weighting}":
                                decorated_structures[spin_angle_tol][weighting].site_properties["p_score"][n["site_to"]],
                            f"site_to_ap_{spin_angle_tol}_{weighting}":
                                decorated_structures[spin_angle_tol][weighting].site_properties["ap_score"][n["site_to"]],
                        })

                row_lists.append(interaction_dict)

    df = pd.DataFrame.from_records(row_lists)
    return df


def get_magnetic_node_information(structure: Structure,
                                  coordination_features: CoordinationFeatures,
                                  spin_angle_tols: tuple[float, ...] = (0.0, 10.0, 20.0),
                                  ) -> pd.DataFrame:
    """
    Extracts node information (local p, ap scores, coordination environment etc.) from magnetic structure and its
    respective CoordinationFeatures object.
    Intended for statistical analysis and as benchmarking of guessing methods, NOT as actual featurization!
    (As magnetic sites are not guessed).
    Note: SimplestChemEnv strategy for constructing CoordinationFeature object is assumed.
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param spin_angle_tols: maximum absolute angles in deg to count two magnetic vectors
    as parallel / antiparallel.
    :return: a structurewise dataframe with each row corresponding to a magnetic site in the structure and
    its corresponding site information.
    """
    decorated_structures = compute_p_ap_decorated_structures(structure=structure,
                                                             coordination_features=coordination_features,
                                                             spin_angle_tols=spin_angle_tols)
    # Store relevant information in dataframe
    row_lists = []
    for site_idx, site in enumerate(structure.sites):
        # Collect node information about all magnetic sites (irrespective of whether neighbors are also magnetic)
        if np.any(site.properties["magmom"].get_moment()):
            site_dict = {"site": site_idx,
                         "site_element": coordination_features.sites.elements[site_idx],
                         "site_oxidation": coordination_features.sites.oxidations[site_idx],
                         "site_ce": [i["ce_symbols"][0] for i in coordination_features.ces if i["site"] == site_idx][0],
                         "site_csm": [i["csms"][0] for i in coordination_features.ces if i["site"] == site_idx][0],
                         "ligand_elements": [coordination_features.sites.elements[d["site_to"]] for d
                                             in coordination_features.distances if d["site"] == site_idx],
                         }
            # Sanity check
            assert int(site_dict["site_ce"].split(":")[1]) == len(site_dict["ligand_elements"])

            for spin_angle_tol in spin_angle_tols:
                for weighting in ["include_ligand_multiplicities", "exclude_ligand_multiplicities"]:
                    site_dict.update({
                        f"site_p_{spin_angle_tol}_{weighting}":
                            decorated_structures[spin_angle_tol][weighting].site_properties["p_score"][site_idx],
                        f"site_ap_{spin_angle_tol}_{weighting}":
                            decorated_structures[spin_angle_tol][weighting].site_properties["ap_score"][site_idx],
                        f"site_collinearity_{spin_angle_tol}_{weighting}":
                            decorated_structures[spin_angle_tol][weighting].site_properties["p_score"][site_idx]
                            + decorated_structures[spin_angle_tol][weighting].site_properties["ap_score"][site_idx]
                    })

            row_lists.append(site_dict)

    df = pd.DataFrame.from_records(row_lists)
    return df


def get_all_edge_information(structure: Structure,
                             coordination_features: CoordinationFeatures,
                             ) -> pd.DataFrame:
    """
    Extract edge information (spin and bond angles, ligand types etc. ) of all connected cation sites from magnetic
    structure and its respective CoordinationFeatures object.
    Intended for statistical analysis and as benchmarking of guessing methods, NOT as actual featurization!
    Intended for analysis of electropositive side groups to rationalize AFM exchange in 90° bond angle situations
    (see Geertsma and Khomskii on CuGeO3, 10.1103/PhysRevB.54.3011).
    Note: SimplestChemEnv strategy for constructing CoordinationFeature object is assumed.
    !Note: as opposed to magnetic edge analysis, the double counting of edges is eliminated here! (as later
    analyses include analysis of neighboring edges, would complicate things).
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :return: pandas.DataFrame, each row represents one edge, with infos on relative spin and bond angles,
    node infos like ces etc.
    """
    # Store relevant information in dataframe
    row_lists = []
    for n in coordination_features.ce_neighbors:
        # Only include connections between site with non-zero magnetic moments
        v1 = structure.sites[n["site"]].properties["magmom"].get_moment()
        v2 = structure.sites[n["site_to"]].properties["magmom"].get_moment()

        interaction_dict = {"idx_site_and_site_to": "_".join([str(idx) for idx in sorted([n["site"], n["site_to"]])]),
                            "site": n["site"],
                            "site_element": coordination_features.sites.elements[n["site"]],
                            "site_oxidation": coordination_features.sites.oxidations[n["site"]],
                            "site_ce": [i["ce_symbols"][0] for i in coordination_features.ces
                                        if i["site"] == n["site"]][0],
                            "site_csm": [i["csms"][0] for i in coordination_features.ces
                                         if i["site"] == n["site"]][0],
                            "site_to": n["site_to"],
                            "site_to_element": coordination_features.sites.elements[n["site_to"]],
                            "site_to_oxidation": coordination_features.sites.oxidations[n["site_to"]],
                            "site_to_ce": [i["ce_symbols"][0] for i in coordination_features.ces
                                            if i["site"] == n["site_to"]][0],
                            "site_to_csm": [i["csms"][0] for i in coordination_features.ces
                                            if i["site"] == n["site_to"]][0],
                            "connectivity": n["connectivity"],
                            "ligand_elements": str(set(
                                [coordination_features.sites.elements[i] for i in n["ligand_indices"]])),
                            "ligand_indices": n["ligand_indices"],
                            "ligand_indices_string": "_".join([str(idx) for idx in sorted(n["ligand_indices"])]),
                            "bond_angles": [round(ang, 6) for ang in n["angles"]],
                            "distance": round(n["distance"], 6)
                            }
        if np.any(v1) and np.any(v2):
            interaction_dict.update({"spin_angle": round(get_angle_between(v1, v2), 6)})
        else:
            interaction_dict.update({"spin_angle": None})
        row_lists.append(interaction_dict)

    df = pd.DataFrame.from_records(row_lists)

    # Handle non-existent ce_neghbors as in 1.473 (non-connected subgraphs)
    if df.empty:
        return df

    # Clean df to only take into account interactions once - eliminate equal rows
    # Issue: multiplicity of edges not always 2 but 2 * n as in entry 0.118 -> df.drop_duplicates not sufficient
    # Assert cleaned df is exactly half of the length of old df
    cleaned_df = df.drop_duplicates(subset=["ligand_indices_string", "idx_site_and_site_to"])
    counter = Counter(list(zip(df["idx_site_and_site_to"].values, df["ligand_indices_string"].values)))
    counter = {k: int(v / 2 - 1) for k, v in counter.items() if v > 2}
    for k, v in counter.items():
        mults = df.loc[(df["idx_site_and_site_to"] == k[0]) & (df["ligand_indices_string"] == k[1])][:1]
        for it in range(v):
            cleaned_df = pd.concat([mults, cleaned_df], ignore_index=True)
    try:
        assert len(cleaned_df) * 2 == len(df)
    except AssertionError:
        print(len(df), len(cleaned_df))
    return cleaned_df


def get_mp_magnetic_edge_information(structure: Structure,
                                     coordination_features: CoordinationFeatures,
                                     magmoms: list[float],
                                     zero_magmom_threshold: float = 0.4,
                                     ) -> pd.DataFrame:
    """
    get_magnetic_edge_information equivalent for MP database (scalar, DFT+U-computed magnetic moments).
    Extract edge information (spin and bond angles, ligand types etc.) of connected magnetic sites from magnetic
    structure and its respective CoordinationFeatures object.
    Intended for statistical analysis and as benchmarking of guessing methods, NOT as actual featurization!
    (As magnetic sites are not guessed).
    Note: SimplestChemEnv strategy for constructing CoordinationFeature object is assumed.
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param magmoms: list of magnetic moments per site (extra to also choose Bader magmoms)
    :param zero_magmom_threshold: threshold below which a magnetic moment is counted as zero.
    :return: pandas.DataFrame, each row represents one edge, with infos on relative spin and bond angles,
    node infos like ces etc. for graph representation of structure filtered for edges where both nodes are magnetic.
    """
    assert len(magmoms) == len(structure)
    for site_idx, site in enumerate(structure.sites):
        assert str(site.specie) == coordination_features.sites.elements[site_idx], \
            "Mismatch structure - coordinationfeatures sites!"
    # Assert that with chosen threshold no anionic magnetic sites exist in structure
    for m_idx, magmom in enumerate(magmoms):
        if abs(magmom) > zero_magmom_threshold:
            assert coordination_features.sites.ions[m_idx] == "cation", (f"{m_idx} {magmom} "
                                                                         f"{coordination_features.sites.ions[m_idx]}")

    # Store relevant information in dataframe
    row_lists = []
    for n in coordination_features.ce_neighbors:
        # Only include connections between site with non-zero magnetic moments
        if abs(magmoms[n["site"]]) > zero_magmom_threshold and abs(magmoms[n["site_to"]]) > zero_magmom_threshold:
            interaction_dict = {"site": n["site"],
                                "site_element": coordination_features.sites.elements[n["site"]],
                                "site_oxidation": coordination_features.sites.oxidations[n["site"]],
                                "site_ce": [i["ce_symbols"][0] for i in coordination_features.ces
                                            if i["site"] == n["site"]][0],
                                "site_csm": [i["csms"][0] for i in coordination_features.ces
                                             if i["site"] == n["site"]][0],
                                "site_to": n["site_to"],
                                "site_to_element": coordination_features.sites.elements[n["site_to"]],
                                "site_to_oxidation": coordination_features.sites.oxidations[n["site_to"]],
                                "site_to_ce": [i["ce_symbols"][0] for i in coordination_features.ces
                                               if i["site"] == n["site_to"]][0],
                                "site_to_csm": [i["csms"][0] for i in coordination_features.ces
                                                if i["site"] == n["site_to"]][0],
                                "spin_angle": 0.0 if magmoms[n["site"]] * magmoms[n["site_to"]] > 0 else 180.0,
                                "connectivity": n["connectivity"],
                                "ligand_elements": [coordination_features.sites.elements[i]
                                                    for i in n["ligand_indices"]],
                                "mag_ligand_mag_angle": n["angles"],
                                "distance": n["distance"]
                                }
            row_lists.append(interaction_dict)

    df = pd.DataFrame.from_records(row_lists)
    return df


def compute_p_ap_decorated_structures(structure: Structure,
                                      coordination_features: CoordinationFeatures,
                                      spin_angle_tols: tuple[float, ...] = (0.0, 10.0),
                                      ) -> dict:
    """
    Compute sitewise p and ap scores of the structure for different angle tolerances and weighting schemes.
    :param structure: a pymatgen Structure object with magmom site properties
    :param coordination_features: the corresponding pycoordinationnet CoordinationFeatures object
    :param spin_angle_tols: maximum absolute angles in deg to count two magnetic vectors
    as parallel / antiparallel.
    :return: a nested dictionary of structures decorated with sitewise p and ap scores for different parameters.
    """
    # Compute sitewise p and ap scores
    decorated_structures = {tol: {} for tol in spin_angle_tols}
    for spin_angle_tol in spin_angle_tols:
        for weighting in ["include_ligand_multiplicities", "exclude_ligand_multiplicities"]:
            decorated_structures[spin_angle_tol].update({weighting: get_sitewise_p_ap_scores(structure=structure,
                                                                                             coordination_features=
                                                                                             coordination_features,
                                                                                             spin_angle_tol=
                                                                                             spin_angle_tol,
                                                                                             weighting=weighting)})
    return decorated_structures


def get_bond_angle_occurrences(df: pd.DataFrame,
                               include_ligand_multiplicity: bool = True,
                               normalize: bool = True,
                               n_lattice_points: int = 1,
                               spin_angle_round: int = -1,
                               bond_angle_round: int = 7):
    """
    Calculate bond angle occurrences as function of relative spin angles
    for one compound.
    :param df: Compound-wise pd.DataFrame as yielded by get_magnetic_edge_information()
    :param include_ligand_multiplicity: whether to count all ligands on edge as 1 each (True) or return
        fractional occurrences that sum up to 1 (False).
    :param normalize: whether to set the sum of all bond angles in the compound to 1.0 (so more complex
     structures are not overrepresented in the analysis).
    :param n_lattice_points: Only applicable if normalize=False. Ensures absolute occurrences of the
     *primitive* magnetic cell are counted although conventional magnetic cells are handled in whole
     data analysis process (instead of explicit conversion to magnetic primitive).
    :param spin_angle_round: decimal place spin angle in deg is rounded.
    :param bond_angle_round: decimal place bond angle in deg is rounded.
    :return: list([spin_angle, bond_angle, occurrence of that angle pair])
    """
    # Sanity check implicit conversion to primitive magnetic cell (although weak test as also multiplicity of sites..)
    assert (len(df) / n_lattice_points).is_integer()

    df["spin_angle"] = df["spin_angle"].apply(lambda x: round(x, spin_angle_round))
    df["mag_ligand_mag_angle"] = df["mag_ligand_mag_angle"].apply(lambda ls: [round(x, bond_angle_round) for x in ls])

    return preprocess_count_and_normalize_occurrences(df=df,
                                                      include_ligand_multiplicity=include_ligand_multiplicity,
                                                      columns_to_keep=["spin_angle", "mag_ligand_mag_angle"],
                                                      normalize=normalize,
                                                      n_lattice_points=n_lattice_points)


def get_node_feature_occurrences(df: pd.DataFrame,
                                 normalize: bool = True,
                                 n_lattice_points: int = 1,
                                 make_y_axis_categorical: bool = False,
                                 col0: str = "site_ce",
                                 col1: str = "site_p_10.0_include_ligand_multiplicities"):
    """
    Return occurrences of a column value pair in a node-wise dataframe.
    :param df: Compoundwise pd.DataFrame as yielded by get_magnetic_node_information
    :param normalize: whether to set the sum of all value pairs in the compound to 1.0 (True) or
        to count absolute occurrences in the magnetic primitive cell (False).
    :param n_lattice_points: Only applicable if normalize=False. Ensures absolute occurrences of the
     *primitive* magnetic cell are counted although conventional magnetic cells are handled in whole
     data analysis process (instead of explicit conversion to magnetic primitive).
    :return: list([col0, col1, occurrence of that column value pair])
    """
    # Sanity check of input
    if len(set(df["site"].values)) != len(df):
        raise ValueError(
            "get_node_feature_occurrences() requires a dataframe as yielded by get_magnetic_node_information!"
        )

    # Sanity check implicit conversion to primitive magnetic cell
    assert (len(df) / n_lattice_points).is_integer()

    pp_df = df.drop(columns=[col for col in df if col not in [col0, col1]])

    if make_y_axis_categorical:
        pp_df[col1] = [P_AP_INTERVAL_MAP[inter] for inter in pd.cut(pp_df[col1], P_AP_INTERVALS)]

    return count_and_normalize_occurrences(ls=[tuple(pair) for pair in pp_df.to_numpy()],
                                           normalize=normalize, n_lattice_points=n_lattice_points)


def get_bond_angle_interval_statistics(df: pd.DataFrame, include_ligand_multiplicity: bool = True,
                                       analyze_column: str = "site_csm", n_lattice_points: int = 1,
                                       bond_angle_interval: (float, float) = (85, 95)) -> list[list]:
    """
    Get absolute occurrences of node and edge properties in specific bond angle intervals.
    Used in 90° AFM and FM subdf analysis of connected TM octahedra.
    :param df: Compoundwise pd.DataFrame as yielded by get_magnetic_edge_information() or subdf of it.
    :param include_ligand_multiplicity:whether to count all ligands / angles on edge as 1 each (True) or return
        fractions of their occurrence in edge (False).
    :param analyze_column: column name of df, which property to analyze.
    :param n_lattice_points: Ensures absolute occurrences of the
    *primitive* magnetic cell are counted although conventional magnetic cells are handled in whole
    data analysis process (instead of explicit conversion to magnetic primitive).
    :param  bond_angle_interval: interval of bond angles in which
    :return: list of [value, occurrence] pairs
    """
    occus = []
    for row_id, row in df.iterrows():
        for bond_angle in row["mag_ligand_mag_angle"]:
            if bond_angle_interval[0] <= bond_angle <= bond_angle_interval[1]:
                if include_ligand_multiplicity:
                    occus.append([round(row[analyze_column], 2), 1 / n_lattice_points])
                else:
                    occus.append(
                        [round(row[analyze_column], 2), 1 / (n_lattice_points * len(row["mag_ligand_mag_angle"]))])
    return occus


def count_and_normalize_occurrences(ls: list[tuple], normalize: bool = True, n_lattice_points: int = 1) -> list:
    """
    General utility function for counting and normalizing value pair occurrences.
    :param ls: list of value pairs as tuples, e.g. [(spin_ang0, bond_ang0), (spin_ang1, bond_ang1), ..]
    :param normalize: Whether to normalize the occurrences dictionary (set sum(all_occurrences) to 1)
    :param n_lattice_points: Only applicable if normalize=False. Ensures absolute occurrences of the
    *primitive* magnetic cell are counted although conventional magnetic cells are handled in whole
    data analysis process (instead of explicit conversion to magnetic primitive).
    :return: normalized occurrences dictionary {x: {y: normalized_occurrence_of_y}}
    """
    normalizer = len(ls) if normalize else n_lattice_points

    occus = Counter(ls)
    occus_list = [[key[0], key[1], value / normalizer] for key, value in occus.items()]

    return occus_list


def preprocess_count_and_normalize_occurrences(df: pd.DataFrame, include_ligand_multiplicity: bool = True,
                                               columns_to_keep: list = ["spin_angle", "mag_ligand_mag_angle"],
                                               normalize: bool = True, n_lattice_points: int = 1) -> list:
    """
    :param df: Compoundwise pd.DataFrame as yielded by get_magnetic_edge_information()
    :param include_ligand_multiplicity: whether to count all ligands / angles on edge as 1 each (True) or return
        fractions of their occurrence in edge (False).
    :param columns_to_keep: columns to keep in returned, preprocessed df.
    :param normalize: Whether to normalize the occurrences dictionary (set sum(all_occurrences) to 1)
    :param n_lattice_points: Only applicable if normalize=False. Ensures absolute occurrences of the
    *primitive* magnetic cell are counted although conventional magnetic cells are handled in whole
    data analysis process (instead of explicit conversion to magnetic primitive).
    :return: normalized occurrences dictionary {x: {y: normalized_occurrence_of_y}}
    """
    assert len(columns_to_keep) == 2
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep + ["mag_ligand_mag_angle"]]
    new_df = df.drop(columns=columns_to_drop, inplace=False)

    new_rows = []
    for row_id, row in new_df.iterrows():
        row_to_dict = row.to_dict()
        for ligand_idx in range(len(row["mag_ligand_mag_angle"])):
            unfolded_row = row_to_dict.copy()
            if "mag_ligand_mag_angle" in columns_to_keep:
                unfolded_row.update({"mag_ligand_mag_angle": row["mag_ligand_mag_angle"][ligand_idx]})

            if include_ligand_multiplicity:
                unfolded_row.update({"occurrence": 1.0})
            else:
                unfolded_row.update({"occurrence": 1 / len(row["mag_ligand_mag_angle"])})

            new_rows.append(unfolded_row)
    occus = [((d[columns_to_keep[0]], d[columns_to_keep[1]]), d["occurrence"]) for d in new_rows]
    occus_counter = Counter()
    for k, o in occus:
        occus_counter.update({k: o})

    normalizer = sum(list(occus_counter.values())) if normalize else n_lattice_points
    return [[key[0], key[1], value / normalizer] for key, value in occus_counter.items()]
