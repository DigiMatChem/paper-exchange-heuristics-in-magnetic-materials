"""
Compare model performances and feature importances of models trained on binned p and ap
scores in MAGNDATA - all structures, MAGNDATA - RE-free structures and MP.
Please note as complete p to ap label mapping in collinear MP dataset, only one target
(p) was trained on, but is used for comparison with both p and ap targets
of partly non-collinear MAGNDATA datasets.
Also compare to models trained without heuristic-related features.
"""
import pandas as pd
import json
import numpy as np
import os
import plotly.graph_objects as go
from scipy.stats import ttest_ind

from utils_kga.models.evaluate_RF_models import (plot_grouped_summed_feature_importances,
                                                 get_heuristic_importance_sum)
from utils_kga.models.evaluate_models import get_mean_baseline_f1_score
from utils_kga.general import pretty_plot


result_dir = "../results/p_ap_results/"
os.makedirs(result_dir, exist_ok=True)
colors = {"all-structs_MAGNDATA": "#00556E", "TM-structs_MAGNDATA": "#0089BA", "MP": "#8C1419"}

def wo_f(i: str):
    return (i.split("__")[0] in ["2025-05-09", "2025-05-11", "2025-05-12"] and "MAGNDATA" in i) or (i.split("__")[0] == "2025-05-26" and "3__MP" in i)

cond_dict = {"": lambda i: ((i.split("__")[0] in ["2025-05-09", "2025-05-07"] and "MAGNDATA" in i)
                            or (i.split("__")[0] == "2025-05-25" and "3__MP" in i))
                           and i[-1].isnumeric(),
            "wobcd": lambda i: wo_f(i) and i.endswith("wobcd"),
             "wob": lambda i: wo_f(i) and i.endswith("wob"),
             "woc": lambda i: wo_f(i) and i.endswith("woc"),
             "wod": lambda i: wo_f(i) and i.endswith("wod")
           }

feat_types = ["perc", "bond_angle", "neighbor_distance", "heuristic"]

mods = ["CV", "hold-out"]
trace_names_ordered = ["p--all-structs_MAGNDATA", "ap--all-structs_MAGNDATA",
                       "p--TM-structs_MAGNDATA", "ap--TM-structs_MAGNDATA", "p--MP"]
overall_f1s = {k: {c: {} for c in cond_dict} for k in mods}
elim_box = {k: {t: go.Figure(layout=go.Layout(title=f"{t} {k}", yaxis=go.layout.YAxis(title=f"F1 test ({k})")))
                for t in trace_names_ordered} for k in mods}
stats_dict = {k: {t: {} for t in trace_names_ordered} for k in mods}

for cond_str, cond in cond_dict.items():
    mlog = pd.read_json("../results/complete_model_id_log_mapping.json").T
    mlog["old_index"] = mlog.index
    mlog = mlog.loc[[i for i in mlog.index.values if cond(i)]]
    mlog.rename(index={i: "__".join(i.split("__")[1:]) for i in mlog.index}, inplace=True)
    mlog["trace_names"] = [i.split("__")[2].split("_")[0] + "--" + i.split("__")[3] for i in mlog.index.values]

    score_dict = {}
    for model_id, row in mlog.iterrows():
        model_id_o = row["old_index"]
        with open(f"../results/{model_id_o}/{model_id_o}.json") as f:
            log = json.load(f)
        score_dict.update({model_id: log})

        assert round(mlog.at[model_id, "f1_test_weighted_or_minor_ext"], 6) == round(
            score_dict[model_id]["test_scores_ext_av"]["f1_test_weighted"], 6), (
                (str(mlog.at[model_id, "f1_test_weighted_or_minor_ext"])) + str(score_dict[model_id]["test_scores_ext_av"]["f1_test_weighted"]) + model_id)


    if cond_str == "":
        imp_figs = {"rf": {}, "shapley": {}}
        for v in feat_types:
            mlog[f"{v}_imps_rf"] = mlog.index.map(
                lambda i: get_heuristic_importance_sum(score_dict[i], imp_type="rf", feat_type=v))
            mlog[f"{v}_imps_shapley"] = mlog.index.map(
                lambda i: get_heuristic_importance_sum(score_dict[i], imp_type="shapley", feat_type=v))

            imp_figs["rf"][v] = go.Figure(
                layout=go.Layout(yaxis=go.layout.YAxis(title=f"RF {v} feature importances")))
            imp_figs["shapley"][v] = go.Figure(
                layout=go.Layout(yaxis=go.layout.YAxis(title=f"Shapley {v} feature importances")))

        for cl in range(3):
            mlog[f"f1_test_cv{cl}"] = mlog.index.map(lambda i: score_dict[i]["test_scores_cv"][f"f1_test_{cl}"])
            mlog[f"f1_test_ext{cl}"] = mlog.index.map(lambda i: score_dict[i]["test_scores_ext_av"][f"f1_test_{cl}"])

        mlog_n = mlog.drop(columns=["old_index", "trace_names", "log_path"])
        mlog_n.to_json(os.path.join(result_dir, f"df_summary_p_ap_models{cond_str}.json"))

        cv_fig = go.Figure(layout=go.Layout(yaxis=go.layout.YAxis(title="F1 test (CV)")))
        ext_fig = go.Figure(layout=go.Layout(yaxis=go.layout.YAxis(title="F1 test (hold-out)")))

    f1_table = pd.DataFrame(index=trace_names_ordered)

    for tn in trace_names_ordered:
        subdf = mlog.loc[mlog["trace_names"] == tn]
        assert len(subdf) == 20, f"{tn} {cond_str} {len(subdf)}"
        color = colors[tn.split("--")[1]]

        f1_table.at[tn, "f1_test_weighted_cv_mean"] = subdf["f1_test_weighted_or_minor_cv"].values.mean()
        f1_table.at[tn, "f1_test_weighted_cv_std"] = subdf["f1_test_weighted_or_minor_cv"].values.std()
        f1_table.at[tn, "f1_test_weighted_ext_mean"] = subdf["f1_test_weighted_or_minor_ext"].values.mean()
        f1_table.at[tn, "f1_test_weighted_ext_std"] = subdf["f1_test_weighted_or_minor_ext"].values.std()

        baseline_f1s = np.array(
            [get_mean_baseline_f1_score(y_true=score_dict[i]["ys_cv"]["y_true"]) for i in subdf.index.values])
        f1_table.at[tn, "f1_baseline_mean"] = baseline_f1s.mean()
        f1_table.at[tn, "diff_f1_cv_baseline_mean"] = (
                f1_table.at[tn, "f1_test_weighted_cv_mean"] - f1_table.at[tn, "f1_baseline_mean"])
        f1_table.at[tn, "diff_f1_ext_baseline_mean"] = (
                f1_table.at[tn, "f1_test_weighted_ext_mean"] - f1_table.at[tn, "f1_baseline_mean"])

        overall_f1s["CV"][cond_str][tn] = f1_table.at[tn, "f1_test_weighted_cv_mean"]
        overall_f1s["hold-out"][cond_str][tn] = f1_table.at[tn, "f1_test_weighted_ext_mean"]
        for k in elim_box:
            suffix = "cv" if k == "CV" else "ext"
            elim_box[k][tn].add_trace(go.Box(
                y=subdf[f"f1_test_weighted_or_minor_{suffix}"], boxmean="sd", boxpoints="all",
                marker_color=color, name=cond_str if cond_str != "" else "full", showlegend=False))

        stats_dict["CV"][tn][cond_str] = subdf["f1_test_weighted_or_minor_cv"].values.tolist()
        stats_dict["hold-out"][tn][cond_str] = subdf["f1_test_weighted_or_minor_ext"].values.tolist()

        if cond_str == "":
            cv_fig.add_trace(
                go.Box(y=subdf["f1_test_weighted_or_minor_cv"], name=tn, boxmean="sd", boxpoints="all", marker_color=color))
            cv_fig.add_trace(go.Box(y=[baseline_f1s.mean()], name=tn, marker_color="grey"))
            ext_fig.add_trace(
                go.Box(y=subdf["f1_test_weighted_or_minor_ext"], name=tn, boxmean="sd", boxpoints="all", marker_color=color))
            ext_fig.add_trace(go.Box(y=[baseline_f1s.mean()], name=tn, marker_color="grey"))

            for imp_type, v_dict in imp_figs.items():
                for v, fig in v_dict.items():
                    fig.add_trace(
                        go.Box(y=subdf[f"{v}_imps_{imp_type}"], name=tn, boxmean="sd", boxpoints="all", marker_color=color))

            model_logs = []
            for row_id in subdf.index.values:
                model_logs.append(score_dict[row_id])
            for imp_type in ["rf", "shapley"]:
                fig = plot_grouped_summed_feature_importances(model_logs=model_logs, imp_type=imp_type)
                fig.write_html(os.path.join(result_dir, f"{tn}_{imp_type}_feature_importances.html"))

    f1_table = f1_table.round(decimals=2)
    f1_table.to_csv(os.path.join(result_dir, f"f1_table{cond_str}.csv"))

    if cond_str == "":
        for fig, fig_str in zip([cv_fig, ext_fig],
                                ["f1_test_cv", "f1_test_external"]):
            fig = pretty_plot(fig)
            fig.write_html(os.path.join(result_dir, f"p_ap_models_{fig_str}.html"))
            fig.update_layout(xaxis=dict(tickfont=dict(size=4)))
            fig.update_layout(showlegend=False, autosize=False, width=700, height=550)
            fig.write_image(os.path.join(result_dir, f"p_ap_models_{fig_str}.pdf"))

        for imp_type, v_dict in imp_figs.items():
            for v, fig in v_dict.items():
                fig = pretty_plot(fig)
                fig.write_html(os.path.join(result_dir, f"p_ap_models_{imp_type}_feature_importances_{v}.html"))
                fig.update_layout(xaxis=dict(tickfont=dict(size=4)))
                fig.update_layout(yaxis_range=[0.0, 0.77])
                fig.update_layout(showlegend=False, autosize=False, width=700, height=550)
                fig.write_image(os.path.join(result_dir, f"p_ap_models_{imp_type}_feature_importances_{v}.pdf"))

# Perform Welch's t-test on F1 test scores with full set of features and without sets of heuristic features
welch_df = pd.DataFrame(index=trace_names_ordered)
for mod, t_dict in stats_dict.items():
    for tn, f1_dict in t_dict.items():
        for test_mode in cond_dict:
            if test_mode != "":
                ttest = ttest_ind(a=f1_dict[""], b=f1_dict[test_mode], equal_var=False, alternative="greater")
                welch_df.at[tn, f"{test_mode}_{mod}_statistic"] = ttest.statistic
                welch_df.at[tn, f"{test_mode}_{mod}_pvalue"] = ttest.pvalue

welch_df.to_json(os.path.join(result_dir, f"welch_tests_p_ap.json"))
welch_df = welch_df.round({col: 4 for col in welch_df.columns if "statistic" in col})
welch_df.T.to_latex(os.path.join(result_dir, f"welch_tests_p_ap.txt"))

# Plot cond_str-dependent differences to full-feature model
for k, c in elim_box.items():
    yrange = [0.52, 0.75] if k == "CV" else [0.39, 0.91]
    for tn, fig in c.items():
        fig = pretty_plot(fig)
        fig.write_html(os.path.join(result_dir, f"box_plots_f1_{k}-{tn}_elim.html"))
        fig.update_layout(yaxis_range=yrange)
        fig.write_image(os.path.join(result_dir, f"box_plots_f1_{k}-{tn}_elim.pdf"))

colors = {"all": "#00556E", "TM": "#0089BA", "MP": "#8C1419"}
for t, t_dict in overall_f1s.items():
    for cond_str, c_dict in t_dict.items():
        if cond_str != "":
            fig = go.Figure(layout=go.Layout(yaxis=go.layout.YAxis(title=f"F1 ({t}, {cond_str}) - F1 ({t}, full)")))
            plot_dict = {k.split("-str")[0]: v-t_dict[""][k] for k, v in c_dict.items() if k != ""}
            fig.add_trace(go.Bar(
                x=list(plot_dict.keys()),
                y=list(plot_dict.values()),
                marker_color=[colors[k.split("--")[1]] for k in plot_dict]
            ))
            fig = pretty_plot(fig)
            fig.update_layout(yaxis_range=[-0.068, 0.015])
            fig.write_image(os.path.join(result_dir, f"f1-{t}_comparison_{cond_str}.pdf"))
