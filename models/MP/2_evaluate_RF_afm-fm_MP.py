"""
Compare model performances and feature importances of models trained on FM / AFM
target of MP.
"""
import pandas as pd
import json
import numpy as np
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import ttest_ind

from utils_kga.models.evaluate_RF_models import (plot_grouped_summed_feature_importances,
                                                 get_heuristic_importance_sum)
from utils_kga.models.evaluate_models import get_mean_baseline_f1_score
from utils_kga.general import pretty_plot


result_dir = "../results/fm-afm_results/"
os.makedirs(result_dir, exist_ok=True)
color = "#8C1419"

def date_f(i: str):
    return i.split("__")[0] == "2025-06-04"


cond_dict = {"": lambda i: date_f(i) and i[-1].isnumeric(),
             "wobcd": lambda i: date_f(i) and i.endswith("wobcd"),
             "wob": lambda i: date_f(i) and i.endswith("wob"),
             "woc": lambda i: date_f(i) and i.endswith("woc"),
             "wod": lambda i: date_f(i) and i.endswith("wod"),
           }
mods = ["CV", "hold-out"]
overall_f1s = {k: {} for k in mods}
elim_box = {k: go.Figure(layout=go.Layout(yaxis=go.layout.YAxis(title=f"F1 test ({k})"))) for k in mods}
feat_types = ["bond_angle", "perc", "neighbor_distance", "heuristic"]

for cond_str, cond in cond_dict.items():
    mlog = pd.read_json("../results/complete_model_id_log_mapping.json").T
    mlog["old_index"] = mlog.index
    mlog = mlog.loc[[i for i in mlog.index.values if cond(i)]]
    mlog.rename(index={i: "__".join(i.split("__")[1:]) for i in mlog.index}, inplace=True)
    assert len(mlog) == 20

    score_dict = {}
    for model_id, row in mlog.iterrows():
        model_id_o = row["old_index"]
        with open(f"../results/{model_id_o}/{model_id_o}.json") as f:
            log = json.load(f)
        score_dict.update({model_id: log})

    overall_f1s["CV"][cond_str] = mlog["f1_test_weighted_or_minor_cv"].values  # Store for later comparison between models w. different feature groups
    overall_f1s["hold-out"][cond_str] = mlog["f1_test_weighted_or_minor_ext"].values
    for k, fig in elim_box.items():
        suffix = "cv" if k == "CV" else "ext"
        fig.add_trace(go.Box(
        y=mlog[f"f1_test_weighted_or_minor_{suffix}"], boxmean='sd', boxpoints="all",
        marker_color=color, name=cond_str if cond_str != "" else "full", showlegend=False))

    if cond_str == "":
        imp_figs = {
            "rf": go.Figure(
                layout=go.Layout(yaxis=go.layout.YAxis(title=f"RF feature importances"))),
            "shapley": go.Figure(
                layout=go.Layout(yaxis=go.layout.YAxis(title=f"Shapley feature importances")))
        }
        for imp_type, fig in imp_figs.items():
            for v in feat_types:
                mlog[f"{v}_imps_{imp_type}"] = mlog.index.map(
                    lambda i: get_heuristic_importance_sum(score_dict[i], imp_type=imp_type, feat_type=v))
                fig.add_trace(go.Box(y=mlog[f"{v}_imps_{imp_type}"], name=v,
                                     boxmean="sd", boxpoints="all", marker_color=color))
        for imp_type, fig in imp_figs.items():
            fig = pretty_plot(fig)
            fig.write_html(os.path.join(result_dir, f"fm-afm_models_{imp_type}_feature_importances.html"))
            fig.update_layout(xaxis=dict(tickfont=dict(size=4)))
            fig.update_layout(yaxis_range=[0.0, 0.77])
            fig.update_layout(showlegend=False, autosize=False, width=700, height=550)
            fig.write_image(os.path.join(result_dir, f"fm-afm_models_{imp_type}_heuristic_feature_importances.pdf"))

        f1_dict = {}
        f1_dict["mean_f1_cv"] = mlog["f1_test_weighted_or_minor_cv"].values.mean()
        f1_dict["std_f1_cv"] = mlog["f1_test_weighted_or_minor_cv"].values.std()
        f1_dict["mean_f1_ext"] = mlog["f1_test_weighted_or_minor_ext"].values.mean()
        f1_dict["std_f1_ext"] = mlog["f1_test_weighted_or_minor_ext"].values.std()
        f1_dict["mean_f1_bl"] = np.array(
            [get_mean_baseline_f1_score(y_true=score_dict[i]["ys_cv"]["y_true"]) for i in mlog.index.values]).mean()
        # baseline same for CV and hold-out test set
        f1_dict["diff_cv"] = f1_dict["mean_f1_cv"] - f1_dict["mean_f1_bl"]
        f1_dict["diff_ext"] = f1_dict["mean_f1_ext"] - f1_dict["mean_f1_bl"]
        with open(os.path.join(result_dir, "f1_dict.json"), "w") as f:
            json.dump(f1_dict, f)

        mlog_n = mlog.drop(columns=["old_index", "log_path"])
        mlog_n.to_json(os.path.join(result_dir, f"df_summary_fm-afm_models{cond_str}.json"))

        columns = ("f1_test_weighted_or_minor_cv", "f1_test_weighted_or_minor_ext")
        fig = make_subplots(rows=2, cols=1,
                            specs=[[{"type": "box"}], [{"type": "box"}]],
                            subplot_titles=columns)
        for f_idx, col in enumerate(columns):
            fig.add_trace(go.Box(x=mlog[col], boxmean="sd", boxpoints="all", marker_color=color, name=col),
                          row=f_idx+1, col=1, )
        fig.update_layout(xaxis=dict(range=[0.42, 0.62]),
                          xaxis2=dict(range=[0.42, 0.62]),
                          showlegend=False)
        fig = pretty_plot(fig)
        fig.write_html(os.path.join(result_dir, f"afm-fm_models_MP_metrics.html"))
        fig.update_layout(yaxis=dict(tickfont=dict(size=9)),
                          yaxis2=dict(tickfont=dict(size=9)),
                          xaxis=dict(tickfont=dict(size=14)),
                          xaxis2=dict(tickfont=dict(size=14)),
                          )
        fig.write_image(os.path.join(result_dir, f"afm-fm_models_MP_metrics.pdf"))

        model_logs = []
        for row_id in mlog.index.values:
            model_logs.append(score_dict[row_id])
        for imp_type in ["rf", "shapley"]:
            fig = plot_grouped_summed_feature_importances(model_logs=model_logs, imp_type=imp_type, summarize_below_perc=1)
            fig.write_html(os.path.join(result_dir, f"{imp_type}_feature_importances.html"))

for k, fig in elim_box.items():
    fig = pretty_plot(fig)
    fig.update_layout(yaxis_range=[0.4, 0.67])
    fig.write_image(os.path.join(result_dir, f"box_plots_f1_{k}_elim.pdf"))

for t, t_dict in overall_f1s.items():
    fig = go.Figure(layout=go.Layout(yaxis=go.layout.YAxis(title=f"F1 ({t}, subset) - F1 ({t}, full)")))
    plot_dict = {k.split("-str")[0]: v.mean() - t_dict[""].mean() for k, v in t_dict.items() if k != ""}
    fig.add_trace(go.Bar(
        x=list(plot_dict.keys()),
        y=list(plot_dict.values()),
        marker_color="#8C1419",
        showlegend=False,
    ))
    fig.update_layout(barmode="overlay")
    fig = pretty_plot(fig)
    fig.update_layout(yaxis_range=[-0.004, 0.0155])
    fig.write_image(os.path.join(result_dir, f"f1-{t}_comparison.pdf"))

# Perform Welch's t-test on F1 test scores with full set of features and without sets of heuristic features
welch_df = pd.DataFrame()
for mod, t_dict in overall_f1s.items():
    for test_mode in t_dict:
        if test_mode != "":
            ttest = ttest_ind(a=t_dict[""].tolist(), b=t_dict[test_mode].tolist(), equal_var=False, alternative="greater")
            welch_df.at[test_mode, f"{mod}_statistic"] = ttest.statistic
            welch_df.at[test_mode, f"{mod}_pvalue"] = ttest.pvalue

welch_df.to_json(os.path.join(result_dir, f"welch_tests_fm-afm.json"))
welch_df = welch_df.round({col: 4 for col in welch_df.columns if "statistic" in col})
welch_df.to_latex(os.path.join(result_dir, f"welch_tests_fm-afm.txt"))
