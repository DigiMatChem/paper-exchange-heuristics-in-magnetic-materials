"""
Calculate n_sampling_per_size (size = n_structures) to yield a sample standard
deviation below 0.01 for all sizes.
"""
import json
from math import floor, sqrt
from monty.json import MontyDecoder
import numpy as np
import pandas as pd
from pymatgen.core import Element

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import get_bond_angle_occurrences
from utils_kga.statistical_analysis.ks_test import compute_ks_weighted


n_sampling_per_size_list = [50, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
step_width = 50
replace = True
ligand_multiplicity_string = "no_ligand multiplicity_included"
datastring = "connected_TM_octahedra"

with open("data/dfs_of_magnetic_edge_information.json") as f:
    dict_all_stats = json.load(f)
all_stats = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats.items()}
for ang_df in all_stats.values():
    ang_df["site_is_tm"] = ang_df["site_element"].apply(lambda el: Element(el).is_transition_metal)
    ang_df["site_to_is_tm"] = ang_df["site_to_element"].apply(lambda el: Element(el).is_transition_metal)
    ang_df["ligand_el_set"] = ang_df["ligand_elements"].apply(lambda ls: set(ls))

# For metadata filtering and computation of occurrences in magnetic primitive cells
with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)

for normalize_bool, normalize_string in zip([False, True], ["absolute_occurrences", "normalized_occurrences"]):
    print(normalize_string)
    all_spin_occus = {}
    for md_id, ang_df in all_stats.items():
        test_df = ang_df.loc[(ang_df["site_is_tm"]) & (ang_df["site_to_is_tm"])]
        test_df["ligand_el_set"] = test_df["ligand_elements"].apply(lambda ls: set(ls))
        test_df = test_df.loc[(test_df["site_ce"] == "O:6") & (test_df["site_to_ce"] == "O:6")]

        if not test_df.empty:
            n_lattice_points = df.at[md_id, "n_lattice_points"]
            occus = get_bond_angle_occurrences(df=test_df,
                                                             include_ligand_multiplicity=False,
                                                             normalize=normalize_bool,
                                                             n_lattice_points=n_lattice_points,
                                                             spin_angle_round=0,
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
        sample_mean_dict = {k: np.mean(np.array(v)) for k, v in p_values.items()}
        print(n_sampling_per_size)
        print(f"sample standard deviation: {sample_standard_deviation_dict}")
        print(f"sample mean: {sample_mean_dict}")
        end = len([True for k, v in sample_standard_deviation_dict.items() if v > 0.01 * sample_mean_dict[k]]) == 0
        print("All variances below 0.01: ", end)
        if end:
            break
    exit(1)