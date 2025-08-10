"""
For all datasets, get confusion matrices (or percentages per structure) of mag. site guessing
as compared to actual mag. site. Evaluate all cationic sites. Normalize confusion matrix fractions
to 1.0 per structure and evaluate statistics over whole datasets.
"""
import json
import pandas as pd
from monty.json import MontyDecoder
from monty.serialization import loadfn
import numpy as np
import plotly.express as px

from utils_kga.featurization.sitewise_coordinationnet_features import (d_f_cation_neighbor_filter,
                                                                       magnetic_cation_neighbor_filter)
from utils_kga.general import pretty_plot


magmom_info = tuple(loadfn("../default_magmoms_uncommented.yaml").keys())

with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    info_df = json.load(f, cls=MontyDecoder)

p_idx = pd.read_json(f"data/250430_full_features_target_p_10.0_excl_binned3_all-structs_MAGNDATA.json").index.tolist()
all_md_guesses = pd.DataFrame(index=p_idx, columns=[])
all_md_guesses["contains_REs"] = all_md_guesses.index.map(lambda i: i in info_df.loc[info_df["contains_REs"]].index.values)
for row_id in all_md_guesses.index:
    mag_structure = json.loads(info_df.at[row_id, "mag_structure"], cls=MontyDecoder)
    cn_feat = json.loads(info_df.at[row_id, "coordination_features"], cls=MontyDecoder)
    for site_filter, site_filter_str in zip([d_f_cation_neighbor_filter, magnetic_cation_neighbor_filter],
                                            ["dfcat", "magcat"]):
        confusion = {"fp": [], "tp": [], "fn": [], "tn": []}
        for site_idx, site in enumerate(mag_structure.sites):
            if cn_feat.sites.ions[site_idx] == "cation":
                guess_mag_bool = site_filter(site_idx=site_idx, coordination_features=cn_feat, magmom_info=magmom_info)
                true_mag_bool = any(site.properties["magmom"].get_moment())
                if guess_mag_bool and true_mag_bool:
                    confusion["tp"].append(site_idx)
                elif not guess_mag_bool and true_mag_bool:
                    confusion["fn"].append(site_idx)
                elif guess_mag_bool and not true_mag_bool:
                    confusion["fp"].append(site_idx)
                else:
                    confusion["tn"].append(site_idx)
        num_confusion_matrix_elements = sum([len(v) for v in confusion.values()])
        for k, v in confusion.items():
            all_md_guesses.at[row_id, f"{k}_fraction_{site_filter_str}"] = len(v) / num_confusion_matrix_elements

all_md_guesses.to_json("data/guessing_method_evaluation_MAGNDATA.json")
tm_md_guesses = all_md_guesses.loc[~all_md_guesses["contains_REs"]]

# MP
zero_magmom_threshold = 0.5  # apply same threshold as in ML and stat. analysis for determining "true" mag. sites
with open("../../statistical_analysis/MP/data_and_plots/df_stable_and_unique_MP_db.json") as f:
    info_df = json.load(f, cls=MontyDecoder)

feat_df = pd.read_json("../MP/data/250525_full_features_target_p_exclude_ligand_multiplicities_binned3_MP.json")
assert len(feat_df) == len(set(feat_df.index.values))
all_mp_guesses = pd.DataFrame(index=feat_df.index.values, columns=[])

for row_id in all_mp_guesses.index:
    mag_structure = json.loads(info_df.at[str(row_id), "structure"], cls=MontyDecoder)
    cn_feat = json.loads(info_df.at[str(row_id), "coordination_features"], cls=MontyDecoder)
    for site_filter, site_filter_str in zip([d_f_cation_neighbor_filter, magnetic_cation_neighbor_filter],
                                            ["dfcat", "magcat"]):
        confusion = {"fp": [], "tp": [], "fn": [], "tn": []}
        for site_idx, site in enumerate(mag_structure.sites):
            if cn_feat.sites.ions[site_idx] == "cation":
                guess_mag_bool = site_filter(site_idx=site_idx, coordination_features=cn_feat, magmom_info=magmom_info)
                true_mag_bool = False if abs(site.properties["magmom"]) <= zero_magmom_threshold else True
                if guess_mag_bool and true_mag_bool:
                    confusion["tp"].append(site_idx)
                elif not guess_mag_bool and true_mag_bool:
                    confusion["fn"].append(site_idx)
                elif guess_mag_bool and not true_mag_bool:
                    confusion["fp"].append(site_idx)
                else:
                    confusion["tn"].append(site_idx)
        num_confusion_matrix_elements = sum([len(v) for v in confusion.values()])
        for k, v in confusion.items():
            all_mp_guesses.at[row_id, f"{k}_fraction_{site_filter_str}"] = len(v) / num_confusion_matrix_elements

all_mp_guesses.to_json("data/guessing_method_evaluation_MP.json")

for df, df_string in zip([all_md_guesses, tm_md_guesses, all_mp_guesses], ["all-MAGNDATA", "RE-free-MAGNDATA", "MP"]):
    print(df_string)
    print("Average values: ")
    for col in df.columns:
        if "fraction" in col:
            print(col, round(df[col].values.mean(), 3), round(df[col].values.std(), 3), round(np.median(df[col].values), 3))
    for site_filter_str in ["dfcat", "magcat"]:
        c = np.array([[round(df[f"tp_fraction_{site_filter_str}"].values.mean(), 3),
                       round(df[f"fp_fraction_{site_filter_str}"].values.mean(), 3)],
                      [round(df[f"fn_fraction_{site_filter_str}"].values.mean(), 3),
                       round(df[f"tn_fraction_{site_filter_str}"].values.mean(), 3)]
                      ])
        fig = px.imshow(c,
                        text_auto=True,
                        labels=dict(x="actual value", y="guessed value", color="av. frac."),
                        x=["positive", "negative"],
                        y=["positive", "negative"],
                        )
        fig = pretty_plot(fig)
        fig.update_layout(font=dict(size=36, family="Arial"))
        fig.update_layout(coloraxis=dict(showscale=False))
        fig.update_coloraxes(cmin=0, cmax=0.6)
        fig.write_image(f"data/average_confusion_matrix_{df_string}_{site_filter_str}.pdf")

