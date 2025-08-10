import json
from monty.json import MontyDecoder, MontyEncoder
from monty.serialization import loadfn
import os
import pandas as pd

from utils_kga.featurization.structurewise_coordinationnet_features import *
from utils_kga.coordination_features import CoordinationFeatures

os.makedirs("data", exist_ok=True)

with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)
df["cryst_structure_oxi_states"] = None
df["cryst_structure_dfcat_sites"] = None
df["cryst_structure_magcat_sites"] = None
failed_cryst_coordinationnet, no_sites_guessed_mag = [], []
magmom_info = tuple(loadfn("../default_magmoms_uncommented.yaml").keys())

all_features = {}
for row_id, row in df.iterrows():
    if row["chosen_one"]:
        structure = json.loads(row["cryst_structure"], cls=MontyDecoder)  # non-magnetic
        try:  # Create non-magnetic CoordinationFeatures object
            cn_feat = CoordinationFeatures().from_structure(structure=structure,
                                                            env_strategy="simple",
                                                            guess_oxidation_states_from_composition=False,
                                                            include_edge_multiplicities=True)
        except (ValueError, TypeError):
            try:
                cn_feat = CoordinationFeatures().from_structure(structure=structure,
                                                                env_strategy="simple",
                                                                guess_oxidation_states_from_composition=True,
                                                                include_edge_multiplicities=True)
            except (ValueError, AssertionError) as e:
                print(f"Creation of non-magn. CoordinationFeatures object failed for {row_id}, "
                      f"falling back to CoordinationFeatures object of magnetic structure "
                      f"and removing magmoms of magnetic structure for featurization.")
                failed_cryst_coordinationnet.append(row_id)
                # Usually avoided as oxidation state assignment may differ for some compounds in different settings
                structure = json.loads(row["mag_structure"], cls=MontyDecoder)
                structure.remove_site_property(property_name="magmom")
                cn_feat = json.loads(row["coordination_features"], cls=MontyDecoder)

        # Store oxidation states of non-magnetic coordinationfeatures object for automatminer featurization
        structure.add_oxidation_state_by_site(cn_feat.sites.oxidations)
        df.at[row_id, "cryst_structure_oxi_states"] = structure

        # Store site indices guessed magnetic for later SCM featurization
        for df_col, ion_filter in zip(["cryst_structure_dfcat_sites", "cryst_structure_magcat_sites"],
                                      [d_f_cation_filter, magnetic_cation_neighbor_filter]):
            nonmag_idx = []
            for site_idx in range(structure.num_sites):
                if not ion_filter(site_idx=site_idx, coordination_features=cn_feat, magmom_info=magmom_info):
                    nonmag_idx.append(site_idx)
            structure_copy = structure.copy()
            structure_copy.remove_sites(nonmag_idx)
            if len(structure_copy.sites) == 0:  # magcat for Re / Pu compounds, fall back to dfcat for these cases
                assert df_col == "cryst_structure_magcat_sites"
                df.at[row_id, df_col] = df.at[row_id, "cryst_structure_dfcat_sites"]
                no_sites_guessed_mag.append(row_id)
                print(f"No sites guessed mag for {row_id} ({structure.formula}), falling back to dfcat instead.")
            else:
                df.at[row_id, df_col] = structure_copy

        all_features.update({row_id: get_structurewise_coordinationnet_features(
            structure=structure,
            coordination_features=cn_feat,
            site_filters=(d_f_cation_filter, magnetic_cation_filter),
            stats=("mean", "min", "max", "std", "median", "most_frequent"),
            magmom_info=magmom_info
        )})
feat_df = pd.DataFrame(all_features).T

for col, content in feat_df.isnull().any().iteritems():
    if content:
        print(col)  # Sanity check for None feature values

feat_df.to_json("data/250226_structure_cn_features.json")
to_remove = [c for c in df.columns if not "cryst_structure" in c]
to_remove.remove("total_moment_per_supercell")
df.drop(columns=to_remove,inplace=True)
with open("data/df_grouped_and_chosen_commensurate_MAGNDATA_additional_nonmag_info.json", "w") as f:
    json.dump(df, f, cls=MontyEncoder)
with open("data/structure_ids_with_failed_cryst_struct_coordinationfeatures.txt", "w") as f:
    f.write(str(failed_cryst_coordinationnet))
with open("data/structure_ids_with_no_sites_guessed_mag.txt", "w") as f:
    f.write(str(no_sites_guessed_mag))
