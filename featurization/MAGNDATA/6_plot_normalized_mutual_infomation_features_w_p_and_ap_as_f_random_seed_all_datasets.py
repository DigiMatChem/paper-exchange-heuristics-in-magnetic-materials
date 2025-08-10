"""
Plot (normalized) mutual information(random seed) of features w. targets p and ap.
Compare MAGNDATA (all structures), MAGNDATA (TM structures), MP.
NMI and MI distributions of
- heuristic feature with the highest NMI / MI mean over all random seeds
- feature with the highest NMI / MI mean over all random seeds
Plot both means in box plots over all train splits as well as train-split-wise box plots
corresponding to different random seeds for (N)MI estimation.
Also store info on features with 20 highest mean (N)MIs.
"""
from collections import Counter
import json
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go

from utils_kga.general import pretty_plot


colors = ["#00556E", "#0089BA", "#8C1419"]
yaxis_range_means = {"nmi": [0.047, 0.152], "mi": [0.128, 0.272]}
yaxis_range_ids = {"nmi": [-0.01, 0.25], "mi": [0.09, 0.335]}

metrics = ["nmi", "mi"]
targets = ["p", "ap"]
db_strings = ["all-structs_MAGNDATA", "TM-structs_MAGNDATA", "MP"]

for metric in metrics:
    top20_dict, best_features, best_heuristic_features = [{db: {t: {} for t in targets} for db in db_strings} for _ in range(3)]
    for db_string in db_strings:
        df_path = "data" if "MAGNDATA" in db_string else "../MP/data/"
        date = "250430" if "MAGNDATA" in db_string else "250525"  # corrected missed data leakage
        for target in targets:
            target_name = f"target_{target}_10.0_excl_binned3" if "MAGNDATA" in db_string \
                else f"target_p_exclude_ligand_multiplicities_binned3"  # direct mapping of p and ap in collinear datasets
            for train_ix in range(20):
                mutual_info = pd.read_json(
                    os.path.join(df_path,
                    f"{date}_NMI_MI_H_{date}_full_features_{target_name}_{db_string}train{train_ix}.json"))

                twenty_largest = mutual_info.nlargest(20, f"{metric}_target_sample_mean")
                top20_dict[db_string][target][train_ix] = twenty_largest.drop(
                    columns=[c for c in twenty_largest.columns if c != f"{metric}_target_sample_mean"]).to_dict()

                for li_idx, li in enumerate([best_features, best_heuristic_features]):
                    best = mutual_info.copy()
                    if li_idx == 1:
                        best.drop(index=[i for i in best.index.values if not ("bond_angle" in i or "_perc" in i or "neighbor_distance" in i)], inplace=True)
                    best = best.loc[
                        best[f"{metric}_target_sample_mean"] == best[f"{metric}_target_sample_mean"].values.max()]
                    assert len(best) == 1
                    li[db_string][target].update({train_ix: best})

            for li, li_string in zip([best_features, best_heuristic_features],
                                     ["best_feature", "best_heuristic_feature"]):
                top20_dict[db_string][target][f"{li_string}_keys"] = dict(
                    Counter([b.index.values[0] for b in li[db_string][target].values()]))
                top20_dict[db_string][target][f"{li_string}_mean_and_std"] = \
                    (np.array([t[f"{metric}_target_sample_mean"].values[0] for t in li[db_string][target].values()]).mean(),
                     np.array([t[f"{metric}_target_sample_mean"].values[0] for t in li[db_string][target].values()]).std())
    with open(f"data/top_20_features_p_ap_{metric}.json", "w") as f:
        json.dump(top20_dict, f)

    # Plot means of best features / best bond angle features in overall boxplots,
    # also add box plots of all NMIs(random seed) per best feature / best bond angle feature
    for db_dict, feature_string in zip([best_features, best_heuristic_features],
                                      ["best_features", "best_heuristic_features"]):
        mean_interactive_fig = go.Figure(layout=go.Layout(title=f"{metric} feature - target {feature_string} means",
                                                          yaxis=go.layout.YAxis(title=metric)))
        mean_static_fig = go.Figure(layout=go.Layout(yaxis=go.layout.YAxis(title=metric)))

        for db_idx, (db_string, target_dict) in enumerate(db_dict.items()):
            for target, i_dict in target_dict.items():
                if db_string == "MP" and target == "ap":
                    continue
                db_fig = go.Figure(layout=go.Layout(title=f"{metric} feature - target {target}",
                                                    yaxis=go.layout.YAxis(title=metric),
                                                    xaxis=go.layout.XAxis(title="train set index")))

                # Add means to overall box plots
                best_means = [best_df[f"{metric}_target_sample_mean"].values[0] for best_df in i_dict.values()]
                assert len(best_means) == 20

                # Create box plots including als NMI(rs) per index
                if feature_string == "best_features":
                    for idx, best_df in i_dict.items():
                        all_vals = best_df.drop(
                            columns=[c for c in best_df.columns if not c.startswith(f"{metric}_target_rs")]).values.flatten()
                        db_fig.add_trace(go.Box(
                            x=[idx for _ in range(len(all_vals))],
                            y=all_vals,
                            boxmean="sd",
                            showlegend=False,
                            quartilemethod="linear",
                            marker_color=colors[db_idx],
                        ))
                    db_fig.update_layout(autosize=False, width=1000, height=500, yaxis_range=yaxis_range_ids[metric],)
                    db_fig = pretty_plot(db_fig, width=2000)
                    db_fig.write_image(f"data/{metric}_{feature_string}_{target}_{db_string}_all_train_sets.pdf")

                xaxis_string = target + db_string[:2] + feature_string.split("_")[1][:2] if "MAGNDATA" in db_string \
                    else "MP" + feature_string.split("_")[1][:2]
                xaxis_string_interactive = f"{target}_{db_string}_{feature_string}_means"

                for fi_idx, fi in enumerate([mean_interactive_fig, mean_static_fig]):
                    xs = xaxis_string if fi_idx == 1 else xaxis_string_interactive
                    fi.add_trace(go.Box(
                        x=[xs for _ in range(len(best_means))],
                        y=best_means,
                        quartilemethod="linear",
                        name=f"{xs} {metric} box plots",
                        marker_color=colors[db_idx],
                        boxmean="sd",
                        boxpoints="all"
                    ),)
                    fi = pretty_plot(fi)

            mean_interactive_fig.update_xaxes(dict(tickfont=dict(size=14)))
            mean_interactive_fig.update_layout(legend=dict(font=dict(size=14, color="black")))
            mean_interactive_fig.write_html(f"data/{metric}_{feature_string}_means.html")

            # Update for static plotting
            mean_static_fig.update_layout(showlegend=False,
                                          title=None,
                                          yaxis_range=yaxis_range_means[metric],
                                          )
            mean_static_fig.write_image(f"data/{metric}_{feature_string}_means.pdf")
