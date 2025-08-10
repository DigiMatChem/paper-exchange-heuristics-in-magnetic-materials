"""
Compute structurewise, binned p and ap scores with spin angle tolerance of 10° (3 bins).
Clean df before finding maximal common subset with other datasets in next step.
"""
import json
from monty.json import MontyDecoder

from utils_kga.data_retrieval_and_preprocessing.get_p_ap_scores import get_p_ap_scores


spin_angle_tols = [10.0]
weightings = ["exclude_ligand_multiplicities"]

with open("data/250226_structure_cn_amm_features_dfcat_magcat.json") as f:
    feat_df = json.load(f, cls=MontyDecoder)
n_old_columns = len(feat_df.columns)
with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    info_df = json.load(f, cls=MontyDecoder)
info_df = info_df.loc[info_df["chosen_one"]]
re_free_structure_ids = info_df.loc[~info_df["contains_REs"]].index.values

for row_id, row in info_df.iterrows():
    struct = json.loads(row["mag_structure"], cls=MontyDecoder)
    cn_feat = json.loads(row["coordination_features"], cls=MontyDecoder)

    for spin_angle_tol in spin_angle_tols:
        for weighting in weightings:
            weighting_string = weighting[:4]
            p, ap = get_p_ap_scores(structure=struct,
                                    coordination_features=cn_feat,
                                    spin_angle_tol=spin_angle_tol,
                                    weighting=weighting)

            for target, target_string in zip([p, ap], ["p", "ap"]):
                base_target_string = f"target_{target_string}_" + str(spin_angle_tol) + "_" + weighting_string
                # feat_df.at[row_id, base_target_string] = target

                if target < 0:  # note for now, but remove in first p ap models
                    feat_df.at[row_id, base_target_string + "_binned3"] = 3
                elif target <= 0.005:
                    feat_df.at[row_id, base_target_string + "_binned3"] = 0
                elif target <= 0.995:
                    feat_df.at[row_id, base_target_string + "_binned3"] = 1
                else:
                    feat_df.at[row_id, base_target_string + "_binned3"] = 2

assert len(feat_df.columns) - 2 == n_old_columns

for dataset in ["all-structs",
                "TM-structs"]:
    for t in ["target_p_10.0_excl_binned3",
              "target_ap_10.0_excl_binned3"]:
        ta_df = feat_df.copy()
        ta_df.drop(columns=[c for c in ta_df.columns if ("target_" in c and c != t)], inplace=True)
        ta_df = ta_df.loc[ta_df[t] < 3]
        if dataset == "TM-structs":
            ta_df = ta_df.loc[ta_df.index.isin(re_free_structure_ids)]

        # Before dropping duplicates, order feature columns alphabetically for finding meaningful minimal common subset
        ta_df = ta_df.reindex(sorted(ta_df.columns), axis=1)
        ta_df = ta_df.T.drop_duplicates().T
        ta_df = ta_df.loc[:, (ta_df != ta_df.iloc[0]).any()]

        ta_df.to_json(
            f"data/250430_full_features_{t}_{dataset}_MAGNDATA.json")
