from collections import Counter
import json
from monty.json import MontyDecoder
import os
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.electronic_structure.core import Magmom

from utils_kga.data_retrieval_and_preprocessing.get_p_ap_scores import *


mp_data_dir = "../MP/data_and_plots"
magndata_data_dir = "../MAGNDATA/data/"

overlap_data_dir = "data"
os.makedirs(overlap_data_dir, exist_ok=True)


# MP imports
with open(os.path.join(mp_data_dir, "dfs_of_magnetic_edge_information_0p5_zero-magmom_threshold.json")) as f:
    dict_all_stats = json.load(f)
all_stats_mp = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats.items()}

for ang_df in all_stats_mp.values():
    ang_df["site_ion"] = ang_df.apply(lambda r: r["site_element"] + str(r["site_oxidation"]), axis=1)

# For metadata filtering and computation of occurrences in magnetic primitive cells
with open(os.path.join(mp_data_dir, "df_stable_and_unique_MP_db.json")) as f:
    df_mp = json.load(f, cls=MontyDecoder)


# MAGNDATA imports
with open(os.path.join(magndata_data_dir, "dfs_of_magnetic_edge_information_include_crystallographic_multiples.json")) as f:
    dict_all_stats = json.load(f)
all_stats_md = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats.items()}

for ang_df in all_stats_md.values():
    ang_df["site_ion"] = ang_df.apply(lambda r: r["site_element"] + str(r["site_oxidation"]), axis=1)

with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df_md = json.load(f, cls=MontyDecoder)


matches = []
key_errors = []


for magndata_id, magndata_row in df_md.iterrows():
    struct_md = json.loads(magndata_row["cryst_structure"], cls=MontyDecoder)

    for mp_id, mp_row in df_mp.iterrows():
        struct_mp = json.loads(mp_row["parent_structure"], cls=MontyDecoder)

        if StructureMatcher().fit(struct_md, struct_mp):

            match_dict = {"magndata_id": magndata_id, "mp_id": mp_id, "magndata_is_chosen_one": magndata_row["chosen_one"]}

            magndata_p, magndata_ap = get_p_ap_scores(structure=json.loads(magndata_row["mag_structure"], cls=MontyDecoder),
                                    coordination_features=json.loads(magndata_row["coordination_features"], cls=MontyDecoder),
                                    spin_angle_tol=10,
                                    weighting="exclude_ligand_multiplicities")
            
            # Consider structures with magnetic sites connected via ligand atoms
            if magndata_p >= 0:
                

                match_dict["magndata_p"] = magndata_p
                match_dict["magndata_ap"] = magndata_ap
                match_dict["magndata_is_collinear"] = round(magndata_p + magndata_ap, 4) == 1

                mp_mag_structure = json.loads(mp_row["structure"], cls=MontyDecoder)

                # Apply same magmom threshold as in overall MP analysis that yields maximum of analyzed structures
                for site_idx, site in enumerate(mp_mag_structure.sites):
                    if abs(site.properties["magmom"]) <= 0.5:
                        site.properties["magmom"] = Magmom(0.0)
                        mp_mag_structure.site_properties["magmom"][site_idx] = Magmom(0.0)
                    else:
                        site.properties["magmom"] = Magmom(site.properties["magmom"])
                        mp_mag_structure.site_properties["magmom"][site_idx] = Magmom(site.properties["magmom"])

                mp_p, mp_ap = get_p_ap_scores(structure=mp_mag_structure,
                                        coordination_features=json.loads(mp_row["coordination_features"], cls=MontyDecoder),
                                        spin_angle_tol=0.0,  # irrelevant as collinear dataset
                                        weighting="exclude_ligand_multiplicities",)
                
                match_dict["mp_p"] = mp_p
                match_dict["mp_ap"] = mp_ap
                
                matches.append(match_dict)
                

match_df = pd.DataFrame(matches)
match_df.to_json(os.path.join(overlap_data_dir, "matches_mp_magndata.json"))
