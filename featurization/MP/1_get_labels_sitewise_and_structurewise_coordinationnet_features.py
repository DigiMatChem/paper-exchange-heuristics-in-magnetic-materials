import json
from monty.json import MontyDecoder, MontyEncoder
from monty.serialization import loadfn
import os
import pandas as pd
from pymatgen.electronic_structure.core import Magmom

from utils_kga.data_retrieval_and_preprocessing.get_p_ap_scores import get_p_ap_scores
from utils_kga.data_retrieval_and_preprocessing.multiples_elimination import get_crystallographic_primitive
from utils_kga.featurization.structurewise_coordinationnet_features import *
from utils_kga.coordination_features import CoordinationFeatures


os.makedirs("data", exist_ok=True)

zero_magmom_threshold = 0.5
weighting_string = "exclude_ligand_multiplicities"
with open("../../statistical_analysis/MP/data_and_plots/df_stable_and_unique_MP_db.json") as f:
    df = json.load(f, cls=MontyDecoder)

df["cryst_structure_oxi_states"] = None
df["cryst_structure_dfcat_sites"] = None
df["cryst_structure_magcat_sites"] = None
target_df = pd.DataFrame()
anionic_magmoms, failed_cryst_coordinationnet, no_sites_guessed_mag_but_df_cat, no_sites_guessed_df_cat = [], [], [], []
magmom_info = tuple(loadfn("../default_magmoms_uncommented.yaml").keys())

all_features = {}
for row_id, row in df.iterrows():
    mag_structure = json.loads(row["structure"], cls=MontyDecoder)

    # Assert that no anionic magnetic sites with defined threshold for same handling as in statistical analysis
    coordination_features = json.loads(row["coordination_features"], cls=MontyDecoder)
    magmoms = row["magmoms"]["vasp"]
    has_am = False
    for m_idx, magmom in enumerate(magmoms):
        if abs(magmom) > zero_magmom_threshold:
            try:
                assert coordination_features.sites.ions[m_idx] == "cation"
            except AssertionError as e:
                has_am = True
                anionic_magmoms.append(row_id)
                break
    if has_am:
        continue

    # Preprocess data: apply zero magmom threshold and convert magnetic moments to vector
    for site_idx, site in enumerate(mag_structure.sites):
        if abs(site.properties["magmom"]) <= zero_magmom_threshold:
            site.properties["magmom"] = Magmom(0.0)
            mag_structure.site_properties["magmom"][site_idx] = Magmom(0.0)
        else:
            site.properties["magmom"] = Magmom(site.properties["magmom"])
            mag_structure.site_properties["magmom"][site_idx] = Magmom(site.properties["magmom"])

    p, ap = get_p_ap_scores(structure=mag_structure,
                            coordination_features=json.loads(row["coordination_features"], cls=MontyDecoder),
                            spin_angle_tol=0.0,  # irrelevant as collinear dataset
                            weighting=weighting_string,)

    for target, target_string in zip([p, ap], ["p", "ap"]):
        base_target_string = f"target_{target_string}" + "_" + weighting_string
        # target_df.at[row_id, base_target_string] = target

        if target < 0:  # note for now, but remove in first p ap models. Leave structs here bec. also learn classic lab.
            target_df.at[row_id, base_target_string + "_binned3"] = 3
        elif target <= 0.005:
            target_df.at[row_id, base_target_string + "_binned3"] = 0
        elif target <= 0.995:
            target_df.at[row_id, base_target_string + "_binned3"] = 1
        else:
            target_df.at[row_id, base_target_string + "_binned3"] = 2

    # Get AFM / FM target with global magmom threshold as in publication
    target_df.at[row_id, "target_classic"] = 1 if sum(row["magmoms"]["vasp"]) > 0.1 else 0

    nonmag_structure = get_crystallographic_primitive(json.loads(row["structure"], cls=MontyDecoder))
    try:  # Create non-magnetic CoordinationFeatures object
        cn_feat = CoordinationFeatures().from_structure(structure=nonmag_structure,
                                                        env_strategy="simple",
                                                        guess_oxidation_states_from_composition=False,
                                                        include_edge_multiplicities=True)
    except (ValueError, TypeError):
        try:
            cn_feat = CoordinationFeatures().from_structure(structure=nonmag_structure,
                                                            env_strategy="simple",
                                                            guess_oxidation_states_from_composition=True,
                                                            include_edge_multiplicities=True)
        except (ValueError, AssertionError) as e:
            print(f"Creation of non-magn. CoordinationFeatures object failed for {row_id}, "
                  f"falling back to CoordinationFeatures object of magnetic structure "
                  f"and removing magmoms of magnetic structure for featurization.")
            failed_cryst_coordinationnet.append(row_id)
            # Usually avoided as oxidation state assignment may differ for some compounds in different settings
            nonmag_structure = json.loads(row["structure"], cls=MontyDecoder)
            nonmag_structure.remove_site_property(property_name="magmom")
            cn_feat = json.loads(row["coordination_features"], cls=MontyDecoder)

    # Store oxidation states of non-magnetic coordinationfeatures object for automatminer featurization
    nonmag_structure.add_oxidation_state_by_site(cn_feat.sites.oxidations)
    df.at[row_id, "cryst_structure_oxi_states"] = nonmag_structure

    # Store site indices guessed magnetic for later SCM featurization
    has_df_cat = True
    for df_col, ion_filter in zip(["cryst_structure_dfcat_sites", "cryst_structure_magcat_sites"],
                                  [d_f_cation_filter, magnetic_cation_neighbor_filter]):
        nonmag_idx = []
        for site_idx in range(nonmag_structure.num_sites):
            if not ion_filter(site_idx=site_idx, coordination_features=cn_feat, magmom_info=magmom_info):
                nonmag_idx.append(site_idx)
        structure_copy = nonmag_structure.copy()
        structure_copy.remove_sites(nonmag_idx)
        if len(structure_copy.sites) == 0:
            try:
                assert df_col == "cryst_structure_magcat_sites"
            except AssertionError:
                # This may happen due to ChemEnv issue of not finding any environments, e.g., in cases where
                # cation is coordinated by sites guessed cationic as well (some nitrogen coordinations)
                print(f"No sites guessed mag for {row_id} ({nonmag_structure.formula}), removing entry.")
                no_sites_guessed_df_cat.append(row_id)
                has_df_cat = False
                break
            df.at[row_id, df_col] = df.at[row_id, "cryst_structure_dfcat_sites"]
            no_sites_guessed_mag_but_df_cat.append(row_id)
            print(f"No sites guessed mag for {row_id} ({nonmag_structure.formula}), falling back to dfcat instead.")
        else:
            df.at[row_id, df_col] = structure_copy
    if not has_df_cat:
        continue

    all_features.update({row_id: get_structurewise_coordinationnet_features(
        structure=nonmag_structure,
        coordination_features=cn_feat,
        site_filters=(d_f_cation_filter, magnetic_cation_filter),
        stats=("mean", "min", "max", "std", "median", "most_frequent"),
        magmom_info=magmom_info
    )})
feat_df = pd.DataFrame(all_features).T
assert len(feat_df) == len(target_df) - len(no_sites_guessed_df_cat)
target_df.drop(index=no_sites_guessed_df_cat, inplace=True)
assert feat_df.index.values.tolist() == target_df.index.values.tolist()
feat_df = pd.merge(feat_df, target_df, left_index=True, right_index=True)

for col, content in feat_df.isnull().any().iteritems():
    if content:
        print(col)  # Sanity check for None feature values

feat_df.to_json("data/250525_structure_cn_features_labels.json")

with open("data/df_stable_MP_db_additional_nonmag_info.json", "w") as f:
    json.dump(df, f, cls=MontyEncoder)
with open("data/structure_ids_with_anionic_magmoms.txt", "w") as f:
    f.write(str(anionic_magmoms))
with open("data/structure_ids_with_failed_cryst_struct_coordinationfeatures.txt", "w") as f:
    f.write(str(failed_cryst_coordinationnet))
with open("data/structure_ids_with_no_sites_guessed_mag_but_with_df_cat.txt", "w") as f:
    f.write(str(no_sites_guessed_mag_but_df_cat))
with open("data/structure_ids_with_no_sites_guessed_df_cat.txt", "w") as f:
    f.write(str(no_sites_guessed_df_cat))
