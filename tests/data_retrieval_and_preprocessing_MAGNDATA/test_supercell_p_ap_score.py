"""
Sanity check to evaluate CoordinationFeatures.ce_neighbors approach:
for each entry of the dataset of 938 crystallographically unique MAGNDATA structures, assert that
its 2x2x2 supercell yield the same p and ap score.
"""
from utils_kga.coordination_features import CoordinationFeatures
from utils_kga.data_retrieval_and_preprocessing.get_p_ap_scores import get_p_ap_scores, get_sitewise_p_ap_scores
import json
import logging
from monty.json import MontyDecoder


logging.basicConfig(filename="test_supercell_p_ap_score_SKIPPED_or_FAILED.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO)


with open("data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)
# Compare crystallographically unique structures of MAGNDATA that are also used in later analyses / ML models
df = df.loc[df["chosen_one"]]

# Compare p, ap scores for 10 parameter combinations
spin_angle_tols = [0.0, 10.0, 20.0, 30.0, 40.0]
weightings = ["exclude_ligand_multiplicities", "include_ligand_multiplicities"]

scaling_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


for overall_idx, (row_id, row) in enumerate(df.iterrows()):
    skipping = False
    struct = json.loads(row["mag_structure"], cls=MontyDecoder)
    super_struct = struct.make_supercell(scaling_matrix=scaling_matrix, in_place=False)
    if row["oxidation_states_origin"] == "BVA":
        cn_feat = CoordinationFeatures().from_structure(struct,
                                                        guess_oxidation_states_from_composition=False,
                                                        include_edge_multiplicities=True)
        super_cn_feat = CoordinationFeatures().from_structure(super_struct,
                                                              guess_oxidation_states_from_composition=False,
                                                              include_edge_multiplicities=True)
    else:
        # Guessing procedure may produce different results, try again if no non-zero oxidation states
        for attempt in range(3):
            try:
                cn_feat = CoordinationFeatures().from_structure(struct,
                                                                guess_oxidation_states_from_composition=True,
                                                                include_edge_multiplicities=True)
                super_cn_feat = CoordinationFeatures().from_structure(super_struct,
                                                                      guess_oxidation_states_from_composition=True,
                                                                      include_edge_multiplicities=True)
                break
            except ValueError as e:  # Failed coordination feature object construction when guessing oxid states
                if attempt == 2:
                    logging.info("------------------------------------------------------ \n"
                                 "SKIPPED: \n"
                                 f"{row_id}, {str(e)}")
                    skipping = True
                    break
        if skipping:
            continue

    failing = False
    for spin_angle_tol in spin_angle_tols:
        if failing:
            break
        for weighting in weightings:
            p, ap = get_p_ap_scores(structure=struct,
                                    coordination_features=cn_feat,
                                    spin_angle_tol=spin_angle_tol,
                                    weighting=weighting)
            super_p, super_ap = get_p_ap_scores(structure=super_struct,
                                                coordination_features=super_cn_feat,
                                                spin_angle_tol=spin_angle_tol,
                                                weighting=weighting)
            if p:
                try:
                    assert round(p, 4) == round(super_p, 4), f"{row_id}, {p}, {super_p}"
                    assert round(ap, 4) == round(super_ap, 4), f"{row_id}, {ap}, {super_ap}"
                except AssertionError as e:
                    cations = [el for site_idx, el in enumerate(cn_feat.sites.elements)
                               if cn_feat.sites.ions[site_idx] == "cation"]
                    supercell_cations = [el for site_idx, el in enumerate(super_cn_feat.sites.elements)
                                         if super_cn_feat.sites.ions[site_idx] == "cation"]
                    anions = [el for site_idx, el in enumerate(cn_feat.sites.elements)
                              if cn_feat.sites.ions[site_idx] == "anion"]
                    supercell_anions = [el for site_idx, el in enumerate(super_cn_feat.sites.elements)
                                        if super_cn_feat.sites.ions[site_idx] == "anion"]
                    logging.info("------------------------------------------------------ \n"
                                 "FAILED: \n"
                                 f"{row_id}, {str(e)}"
                                 f"p: {p}, ap: {ap}, super_p: {super_p}, super_ap: {super_ap}"
                                 f"cations: {cations}"
                                 f"supercell_cations: {supercell_cations}"
                                 f"anions: {anions}"
                                 f"supercell_anions: {supercell_anions}"
                                 f"{get_sitewise_p_ap_scores(structure=struct, coordination_features=cn_feat)}"
                                 f"{get_sitewise_p_ap_scores(structure=super_struct, coordination_features=super_cn_feat)}")

                    failing = True
                    break

    if overall_idx % 50 == 0 and overall_idx > 0:
        logging.info("------------------------------------------------------ \n"
                     f"Tested {overall_idx + 1} / {len(df)} structures so far.")
