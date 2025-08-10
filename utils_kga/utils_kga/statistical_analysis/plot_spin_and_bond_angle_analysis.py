import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .get_spin_and_bond_angle_statistics import get_node_feature_occurrences


def plot_cn_ce_sunburst_categorical(stats_dict: dict,
                                all_df: pd.DataFrame,
                                target: str,
                                threshold: float,
                                normalize: bool = True,
                                summarize_below: float = 0.0) -> dict:
    occus_list = []
    o_sum = 0
    for md_id, node_df in stats_dict.items():
        ol = (get_node_feature_occurrences(df=node_df,
                                           normalize=normalize,
                                           col0="site_ce",
                                           col1=target,
                                           n_lattice_points=all_df.at[md_id, "n_lattice_points"],
                                           make_y_axis_categorical=False,
                                                          ))
        o_sum += sum([o[2] for o in ol])
        occus_list.extend(ol)
    print(o_sum)
    occus_df = pd.DataFrame(columns=["site_ce", target, "occurrence"], data=occus_list)
    coll_df = occus_df.loc[occus_df[target]>=0]  # only consider those with at least partial magnetic network / not isolated
    figs = {}
    for condition, condition_string in zip([coll_df[target]>=0,
                                            coll_df[target]<threshold,
                                            coll_df[target]>=threshold],
                                           (f"df[{target}]>=0",
                                            f"df[{target}]<{threshold}",
                                            f"df[{target}]>={threshold}")):
        sub_df = coll_df.loc[condition]
        summed_occus = {}
        for x_val in set(sub_df["site_ce"].values):
            summed_occus[x_val] = np.sum(sub_df.loc[sub_df["site_ce"]==x_val]["occurrence"].values)
        summed_subdf = pd.DataFrame.from_dict(summed_occus, orient="index",  columns=["occurrence"])
        summed_subdf["site_ce"] = summed_subdf.index
        print(np.sum(summed_subdf["occurrence"].values))
        fig = plot_ce_and_cn_fractions(df=summed_subdf, summarize_below=summarize_below)
        fig.update_layout(title=condition_string, font_family="Arial")
        fig.update_layout(paper_bgcolor="rgba(255, 255, 255, 1)")
        fig.update_traces(textfont=dict(family=["Arial" for _ in range(len(fig.data[0]["ids"]))],
                                        size=[28 for _ in range(len(fig.data[0]["ids"]))]))
        fig.show()
        figs[condition_string] = fig
    return figs


def plot_ce_and_cn_fractions(df: pd.DataFrame, summarize_below: float = 0.0) -> go.Figure:
    """
    Return sunburst chart of cn and ce occurrences per sub dataset.
    :param df: pd.DataFrame obj. w. columns "ce", "occurrence"
    :param summarize_below: CN (not ce!) fraction below which entries are summarized as "other"
    :return: a plotly.graph_objs.Figure object
    """
    # Create cn column
    df["site_cn"] = df["site_ce"].apply(lambda ce: ce.split(":")[1])

    # Group low CN occurrences
    summarize_below = summarize_below * df["occurrence"].values.sum()
    cn_occus = {}
    for cn in set(df["site_cn"].values):
        cn_occus[cn] = df.loc[df["site_cn"] == cn]["occurrence"].values.sum()
    df["cn_occurrence"] = df["site_cn"].apply(lambda cn: float(cn_occus[cn]))
    others = pd.DataFrame.from_records([{"site_cn": "other",
                                         "occurrence": df.loc[
                                         df["cn_occurrence"] < summarize_below]["occurrence"].values.sum(),
                                         "site_ce": "--",
                                         "cn_occurrence": df.loc[
                                         df["cn_occurrence"] < summarize_below]["occurrence"].values.sum()}])
    new_df = pd.concat([others, df.loc[df["cn_occurrence"] >= summarize_below]])
    color_discrete_map = dict(zip([str(i) for i in range(1, 25)], reversed(px.colors.qualitative.Light24)))
    color_discrete_map.update({"8": "#8B1811",
                               "6": "#009FE3",
                               "4": "#055268",
                               "5": "#5C9034",
                               "other": "#E49B00"
                               })

    fig = px.sunburst(new_df, path=["site_cn", "site_ce"], values="occurrence",
                      color="site_cn", color_discrete_map=color_discrete_map)
    fig.data[0].textinfo = "label+percent entry"
    fig.data[0].insidetextorientation = "horizontal"
    return fig
