"""
The parallelity score p depends on the definition of neighbors in crystals.
Here, its dependence on the analysis parameters of the SimplestChemEnv strategy is
determined for the dataset of 938 cryst. unique MAGNDATA structures.
"""
import json
from monty.json import MontyDecoder
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy

from utils_kga.coordination_features import CoordinationFeatures
from utils_kga.data_retrieval_and_preprocessing.get_p_ap_scores import get_p_ap_scores
from utils_kga.general import pretty_plot


with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)
df = df.loc[df["chosen_one"]]

distance_cutoffs = [1.2, 1.4, 1.6]
angle_cutoffs = [0.1, 0.3, 0.5]
p_score_dict = {}
failed_cn_feat_dict = {d: {a: [] for a in angle_cutoffs} for d in distance_cutoffs}

for row_id, row in df.iterrows():
    structure = json.loads(row["mag_structure"], cls=MontyDecoder)
    structurewise_p_score_dict = {d: {} for d in distance_cutoffs}
    structurewise_scores = []
    for d in distance_cutoffs:
        for a in angle_cutoffs:
            if d == 1.4 and a == 0.3:
                cn_feat = json.loads(row["coordination_features"], cls=MontyDecoder)
            else:
                try:
                    cn_feat = CoordinationFeatures().from_structure(structure=structure,
                                                                    env_strategy=SimplestChemenvStrategy(
                                                                        distance_cutoff=d, angle_cutoff=a),
                                                                    guess_oxidation_states_from_composition=False,
                                                                    include_edge_multiplicities=True)
                except (ValueError, TypeError):
                    try:
                        cn_feat = CoordinationFeatures().from_structure(structure=structure,
                                                                        env_strategy=SimplestChemenvStrategy(
                                                                            distance_cutoff=d, angle_cutoff=a),
                                                                        guess_oxidation_states_from_composition=True,
                                                                        include_edge_multiplicities=True)
                    except (ValueError, AssertionError) as e:
                        failed_cn_feat_dict[d][a].append(row_id)
                        continue

            p, ap = get_p_ap_scores(structure=structure,
                                    coordination_features=cn_feat,
                                    spin_angle_tol=10.0,  # param combi as for ML models
                                    weighting="exclude_ligand_multiplicities")  # param combi as for ML models
            structurewise_p_score_dict[d][a] = p
            structurewise_scores.append(p)
    p_score_dict[row_id] = structurewise_p_score_dict
    p_a = np.array([p for p in structurewise_scores if p >= 0])
    print(row_id, np.mean(p_a), np.std(p_a))

with open("p_as_f_chemenv_params.json", "w") as f:
    json.dump(p_score_dict, f)
with open("p_as_f_chemenv_params_failures.json", "w") as f:
    json.dump(failed_cn_feat_dict, f)

def get_label_binned(val):
    if val < 0:
        return 3
    elif val <= 0.05:
        return 0
    elif val <= 0.95:
        return 1
    return 2

with open("p_as_f_chemenv_params.json") as f:
    p_score_dict = json.load(f)

p_abs_dev = {str(d): {str(a): [] for a in angle_cutoffs} for d in distance_cutoffs}
p_different_label = {str(d): {str(a): [] for a in angle_cutoffs} for d in distance_cutoffs}
for md_id, d_dict in p_score_dict.items():
    ref = d_dict["1.4"]["0.3"]  # ChemEnv params used in stat. analysis, labeling and featurization
    ref_binned = get_label_binned(ref)
    for d, a_dict in d_dict.items():
        for a, p in a_dict.items():
            if (p < 0 and ref < 0) or (p >= 0 and ref >= 0):
                p_abs_dev[d][a].append(round(abs(p - ref), 6))
                p_binned = get_label_binned(p)
                if p_binned != ref_binned:
                    p_different_label[d][a].append(md_id)
            else:
                p_different_label[d][a].append(md_id)

p_abs_dev_mean = {d: {a: np.mean(np.array(p)) for a, p in a_dict.items()} for d, a_dict in p_abs_dev.items()}
p_abs_dev_median = {d: {a: np.median(np.array(p)) for a, p in a_dict.items()} for d, a_dict in p_abs_dev.items()}
p_perc_iso = {d: {a: 1-len(p)/len(p_score_dict) for a, p in a_dict.items()} for d, a_dict in p_abs_dev.items()}
p_diff_label = {d: {a: len(p)/len(p_score_dict) for a, p in a_dict.items()} for d, a_dict in p_different_label.items()}

for m, m_string in zip([p_abs_dev_mean, p_abs_dev_median, p_perc_iso, p_diff_label,],
                       ["mean_abs_dev", "median_abs_dev", "perc_disagree_iso",
                        "perc_different_label"]):
    m_df = pd.DataFrame.from_dict(m, orient="index")
    m_df = m_df.round(decimals=4)
    fig = px.imshow(m_df, text_auto=True)
    fig.update_layout(font=dict(size=24, family="Arial"))
    fig = pretty_plot(fig)
    fig.write_image(f"p_score_dependence_on_chemenv_param_{m_string}.pdf")
