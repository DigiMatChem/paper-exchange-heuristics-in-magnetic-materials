"""
Calculate n_sampling_per_size (size = n_structures) to yield a sample standard
deviation below 0.01 for all sizes.
"""
import json
from math import floor, sqrt
from monty.json import MontyDecoder
import numpy as np
import os
import pandas as pd
from pymatgen.core import Element

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import (get_mp_magnetic_edge_information,
                                                                               get_bond_angle_occurrences)
from utils_kga.statistical_analysis.ks_test import compute_ks_weighted


zero_magmom_threshold = 0.5
n_sampling_per_size_list = [50, 200, 500, 1000, 2000, 10000, 20000, 50000]
step_width = 50
replace = True
ligand_multiplicity_string = "no_ligand multiplicity_included"
datastring = "connected_TM_octahedra"

with open("data/unique_4559_cnfeat_MP_db_from_api.json") as f:
    df = json.load(f, cls=MontyDecoder)

if not os.path.isfile("data/dfs_of_magnetic_edge_information_0p5_zero-magmom_threshold.json"):
    all_stats_dict = {}
    for row_id, row in df.iterrows():
        structure = row["structure_dict"]
        coordination_features = json.loads(row["coordination_features"], cls=MontyDecoder)
        try:
            edge_result = get_mp_magnetic_edge_information(structure=structure,
                                                           coordination_features=coordination_features,
                                                           magmoms=structure.site_properties["magmom"],
                                                           zero_magmom_threshold=zero_magmom_threshold)
        except AssertionError as e:
            continue

        if not edge_result.empty:  # This can be empty in the case of isolated magnetic sites
            # edge_result["ligand_el_set"] = edge_result["ligand_elements"].apply(lambda ls: set(ls))
            edge_result["site_is_tm"] = edge_result["site_element"].apply(lambda el: Element(el).is_transition_metal)
            edge_result["site_to_is_tm"] = edge_result["site_to_element"].apply(lambda el: Element(el).is_transition_metal)
            all_stats_dict[row_id] = edge_result

    with open(f"data/dfs_of_magnetic_edge_information_0p5_zero-magmom_threshold.json", "w") as f:
        json.dump({key: df.to_dict() for key, df in all_stats_dict.items()}, f)

# Load edge-df
with open("data/dfs_of_magnetic_edge_information_0p5_zero-magmom_threshold.json") as f:
    dict_all_stats = json.load(f)
all_stats = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats.items()}


for normalize_bool, normalize_string in zip([False, True], ["absolute_occurrences", "normalized_occurrences"]):
    print(normalize_string)
    all_spin_occus = {}
    for md_id, ang_df in all_stats.items():
        test_df = ang_df.loc[(ang_df["site_is_tm"]) & (ang_df["site_to_is_tm"])]
        test_df["ligand_el_set"] = test_df["ligand_elements"].apply(lambda ls: set(ls))
        test_df = test_df.loc[(test_df["site_ce"] == "O:6") & (test_df["site_to_ce"] == "O:6")]

        if not test_df.empty:
            n_lattice_points = df.at[md_id, "n_mag_lattice_points"]
            occus = get_bond_angle_occurrences(df=test_df,
                                               include_ligand_multiplicity=False,
                                               normalize=normalize_bool,
                                               n_lattice_points=n_lattice_points,
                                               spin_angle_round=-1,
                                               bond_angle_round=7)
            all_spin_occus[md_id] = occus

    for n_sampling_per_size in n_sampling_per_size_list:
        sample_sizes = [i * step_width for i in range(1, floor(len(all_spin_occus) / step_width) + 1)] + [
            len(all_spin_occus)]
        p_values = {s: [] for s in sample_sizes}
        for sample_size in sample_sizes:
            for _ in range(n_sampling_per_size):
                some_spin_occus_ids = np.random.choice(list(all_spin_occus.keys()), size=sample_size, replace=replace)

                some_spin_occus = []
                for selected_id in some_spin_occus_ids:
                    some_spin_occus.extend(all_spin_occus[selected_id])
                # Compute KS Test
                fm_occus = [ls for ls in some_spin_occus if ls[0] <= 10.0]
                afm_occus = [ls for ls in some_spin_occus if ls[0] >= 170.0]

                weighted_ks_test_d, weighted_ks_test_p = compute_ks_weighted(afm_occus, fm_occus)
                p_values[sample_size].append(weighted_ks_test_p)

        sample_standard_deviation_dict = {k: np.std(np.array(v)) / sqrt(len(v)) for k, v in p_values.items()}
        print(n_sampling_per_size)
        print(f"sample standard deviation: {sample_standard_deviation_dict}")
        end = len([True for v in sample_standard_deviation_dict.values() if v > 0.01]) == 0
        print("All sample standard deviations below 0.01: ", end)
        if end:
            break
