"""
Plot (normalized) mutual information(random seed) of features w. FM / AFM classification target of the MP database.
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


df_path = "data"
target_name ="target_classic"
target_str = "FM / AFM"
mp_color = "#8C1419"
metrics = ["nmi", "mi"]

for metric in metrics:
    top20_dict, best_features, best_heuristic_features = [{} for _ in range(3)]
    for train_ix in range(20):
        mutual_info = pd.read_json(
            os.path.join(df_path, f"250525_NMI_MI_H_250525_full_features_{target_name}_MPtrain{train_ix}.json"))

        twenty_largest = mutual_info.nlargest(20, f"{metric}_target_sample_mean")
        top20_dict[train_ix] = twenty_largest.drop(
            columns=[c for c in twenty_largest.columns if c != f"{metric}_target_sample_mean"]).to_dict()

        for li_idx, li in enumerate([best_features, best_heuristic_features]):
            best = mutual_info.copy()
            if li_idx == 1:
                best.drop(index=[i for i in best.index.values if not ("bond_angle" in i or "_perc" in i or "neighbor_distance" in i)], inplace=True)
            best = best.loc[
                best[f"{metric}_target_sample_mean"] == best[f"{metric}_target_sample_mean"].values.max()]
            assert len(best) == 1
            li.update({train_ix: best})

    for li, li_string in zip([best_features, best_heuristic_features],
                             ["best_feature", "best_heuristic_feature"]):
        top20_dict[f"{li_string}_keys"] = dict(
            Counter([b.index.values[0] for b in li.values()]))
        top20_dict[f"{li_string}_mean_and_std"] = \
            (np.array([t[f"{metric}_target_sample_mean"].values[0] for t in li.values()]).mean(),
             np.array([t[f"{metric}_target_sample_mean"].values[0] for t in li.values()]).std())
    with open(f"data/250525_top_20_features_fm-afm_{metric}.json", "w") as f:
        json.dump(top20_dict, f)

    # Plot means of best features / best bond angle features in overall boxplots,
    # also add box plots of all NMIs(random seed) per best feature / best bond angle feature
    mean_interactive_fig = go.Figure(layout=go.Layout(title=f"{metric} feature - target {target_str} best means",
                                                      yaxis=go.layout.YAxis(title=metric)))
    mean_static_fig = go.Figure(layout=go.Layout(yaxis=go.layout.YAxis(title=metric)))
    db_fig = go.Figure(layout=go.Layout(title=f"{metric} feature - target {target_str} all train sets",
                                        yaxis=go.layout.YAxis(title=metric),
                                        xaxis=go.layout.XAxis(title="train set index")))

    for i_dict, feature_string in zip([best_features, best_heuristic_features],
                                      ["best_features", "best_heuristic_features"]):

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
                    marker_color=mp_color,
                ))
            db_fig.update_layout(autosize=False, width=1000, height=500,)
            db_fig = pretty_plot(db_fig, width=2000)
            db_fig.write_image(f"data/250525_{metric}_{feature_string}_{target_name}_MP_all_train_sets.pdf")

        xaxis_string = "MP" + feature_string.split("_")[1][:2]
        xaxis_string_interactive = f"MP_{feature_string}_means"

        for fi_idx, fi in enumerate([mean_interactive_fig, mean_static_fig]):
            xs = xaxis_string if fi_idx == 1 else xaxis_string_interactive
            fi.add_trace(go.Box(
                x=[xs for _ in range(len(best_means))],
                y=best_means,
                quartilemethod="linear",
                name=f"{xs} {metric} box plots (linear)",
                marker_color=mp_color,
                boxmean="sd",
                boxpoints="all"
            ),)
            fi = pretty_plot(fi)

    mean_interactive_fig.update_xaxes(dict(tickfont=dict(size=14)))
    mean_interactive_fig.update_layout(legend=dict(font=dict(size=14, color="black")))
    mean_interactive_fig.write_html(f"data/250525_{metric}_{target_name}_means.html")

    # Update for static plotting
    mean_static_fig.update_layout(showlegend=False,
                                  title=None,
                                  )
    mean_static_fig.write_image(f"data/250525_{metric}_{target_name}_means.pdf")
