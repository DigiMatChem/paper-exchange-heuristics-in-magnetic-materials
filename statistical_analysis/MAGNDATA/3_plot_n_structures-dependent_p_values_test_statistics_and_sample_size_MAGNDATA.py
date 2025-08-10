"""
Repeatedly perform KS test on bootstrapped subsamples of full dataset
to investigate dependence of test statistics d, p values and (m * o) / (m + o) of KS test on sample size.
Note: bootstrap structures, not single bond angle occurrences, as the latter would not be physically meaningful.
Output: interactive figure of p and d boxplots as f(n_structures),
individual static plots of p, d and (m * o) / (m + o) as f(n_structures).
"""
import json
from math import floor
from monty.json import MontyDecoder
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.core import Element

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import get_bond_angle_occurrences
from utils_kga.statistical_analysis.ks_test import compute_ks_weighted
from utils_kga.general import pretty_plot

# Determined in 2_get_n_structures-dependent_p_value_MAGNDATA_determine_n_sampling_per_size.py and
# ../MP/2_get_n_structures-dependent_p_value_MP_determine_n_sampling_per_size.py as minimum
# value to yield sample standard deviation < 0.01 for all test parameter combinations
n_sampling_per_size = 1000
step_width = 50
replace = True
datastring = "connected_TM_octahedra"
ligand_multiplicity_bool = False
ligand_multiplicity_string = "no_ligand multiplicity_included"

plot_dir = "plots/sample-size-dependent-p-value"
os.makedirs(plot_dir, exist_ok=True)

with open("data/dfs_of_magnetic_edge_information.json") as f:
    dict_all_stats = json.load(f)
all_stats = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats.items()}
for ang_df in all_stats.values():
    ang_df["site_is_tm"] = ang_df["site_element"].apply(lambda el: Element(el).is_transition_metal)
    ang_df["site_to_is_tm"] = ang_df["site_to_element"].apply(lambda el: Element(el).is_transition_metal)

# For metadata filtering and computation of occurrences in magnetic primitive cells
with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)

for normalize_bool, normalize_string in zip([False, True], ["absolute_occurrences", "normalized_occurrences"]):
    interactive_fig = make_subplots(specs=[[{"secondary_y": True}]])
    all_spin_occus = {}
    for md_id, ang_df in all_stats.items():
        test_df = ang_df.loc[(ang_df["site_is_tm"]) & (ang_df["site_to_is_tm"])]
        test_df["ligand_el_set"] = test_df["ligand_elements"].apply(lambda ls: set(ls))
        test_df = test_df.loc[(test_df["site_ce"] == "O:6") & (test_df["site_to_ce"] == "O:6")]

        if not test_df.empty:
            n_lattice_points = df.at[md_id, "n_lattice_points"]
            occus = get_bond_angle_occurrences(df=test_df,
                                                             include_ligand_multiplicity=ligand_multiplicity_bool,
                                                             normalize=normalize_bool,
                                                             n_lattice_points=n_lattice_points,
                                                             spin_angle_round=0,
                                                             bond_angle_round=7)
            all_spin_occus[md_id] = occus
    sample_sizes = [i * step_width for i in range(1, floor(len(all_spin_occus)/step_width) + 1)] + [len(all_spin_occus)]
    p_values = {s: [] for s in sample_sizes}
    d_values = {s: [] for s in sample_sizes}
    ks_sizes = {s: [] for s in sample_sizes}  # track m * n / (m + n)
    for sample_size in sample_sizes:
        for _ in range(n_sampling_per_size):
            some_spin_occus_ids = np.random.choice(list(all_spin_occus.keys()), size=sample_size, replace=replace)

            some_spin_occus = []
            for selected_id in some_spin_occus_ids:
                some_spin_occus.extend(all_spin_occus[selected_id])
            # Compute KS Test
            fm_occus = [ls for ls in some_spin_occus if ls[0] <= 10.0]
            afm_occus = [ls for ls in some_spin_occus if ls[0] >= 170.0]

            weighted_ks_test_d, weighted_ks_test_p = compute_ks_weighted(afm_occus, fm_occus)
            p_values[sample_size].append(weighted_ks_test_p)
            d_values[sample_size].append(weighted_ks_test_d)
            m = np.array([ls[2] for ls in afm_occus]).sum()
            n = np.array([ls[2] for ls in fm_occus]).sum()
            ks_sizes[sample_size].append((m * n) / (m + n))

    interactive_fig = make_subplots(specs=[[{"secondary_y": True}]])
    interactive_fig.add_trace(go.Scatter(
            x=[0, sample_sizes[-1] + 10],
            y=[0.01, 0.01],
            mode="lines",
            marker_color="black",
            name="0.01 significance threshold",
            marker_opacity=0.1,
            line=dict(dash="dash"),
        ),
        secondary_y=False
    )
    for measure, measure_string, color, sec_y in zip([p_values, d_values],
                                                     ["p", "d"],
                                                     ["red", "green"],
                                                     [False, True]
                                                     ):
        mean_p = {k: np.mean(np.array(v)) for k, v in measure.items()}

        interactive_fig.add_trace(go.Scatter(
            x=list(mean_p.keys()),
            y=list(mean_p.values()),
            marker_color=color,
            name=f"{measure_string} mean",
        ),
        secondary_y=sec_y)

        box_x, box_y = [], []
        for sample_size, vals in measure.items():
            box_x.extend([sample_size] * len(vals))
            box_y.extend(vals)
        interactive_fig.add_trace(go.Box(
            x=box_x,
            y=box_y,
            quartilemethod="linear",
            name=f"{measure_string} box plots (linear)",
            marker_color=color,
        ),
        secondary_y=sec_y)

    interactive_fig = pretty_plot(interactive_fig)
    interactive_fig.update_layout(titlefont=dict(size=12), title_text=f"MAGNDATA: dependency of p value and test "
                                                                      f"statistic d as f(n structures)  ({datastring}, "
                                                                      f"{ligand_multiplicity_string}, "
                                                                      f"{normalize_string})")

    interactive_fig.update_xaxes(title_text="n structures")
    interactive_fig.update_layout(xaxis_range=[0, sample_sizes[-1] + 10])
    interactive_fig.update_yaxes(title_text="p value", secondary_y=False)
    interactive_fig.update_yaxes(title_text="d", secondary_y=True)
    interactive_fig.write_html(os.path.join(plot_dir, f"MAGNDATA_p_and_d_value_as_f_n_structures_{datastring}_"
                                                      f"{ligand_multiplicity_string}_{normalize_string}.html"))

    for measure, measure_string in zip([p_values, d_values, ks_sizes], ["p", "d", "(m * o) / (m + o)"]):
        static_fig = go.Figure(layout=go.Layout(xaxis=go.layout.XAxis(title="n structures"),
                                                       yaxis=go.layout.YAxis(title=measure_string),
                                                       title=f"MAGNDATA: {measure_string} dependency as "
                                                             f"f(n structs) ({datastring}, "
                                                             f"{ligand_multiplicity_string}, {normalize_string})"))
        if measure == p_values:
            static_fig.add_trace(go.Scatter(
                x=[0, sample_sizes[-1] + 10],
                y=[0.01, 0.01],
                mode="lines",
                marker_color="black",
                name="0.01 significance threshold",
                marker_opacity=0.1,
                line=dict(dash="dash"),
                showlegend=False,
        ))

        mean = {k: np.mean(np.array(v)) for k, v in measure.items()}

        static_fig.add_trace(go.Scatter(
            x=list(mean.keys()),
            y=list(mean.values()),
            marker_color="red",
            name=f"{measure_string} mean",
            showlegend=False,
        ))

        box_x, box_y = [], []
        for sample_size, vals in measure.items():
            box_x.extend([sample_size] * len(vals))
            box_y.extend(vals)
        static_fig.add_trace(go.Box(
            x=box_x,
            y=box_y,
            quartilemethod="linear",
            name=f"{measure_string} box plots (linear)",
            marker_color="red",
            showlegend=False,
        ))

        if measure == ks_sizes:
            measure_string = "ks_test_size"
        static_fig = pretty_plot(static_fig, width=2000)
        static_fig.update_layout(title=dict(font=dict(size=10)))
        static_fig.write_image(os.path.join(plot_dir, f"MAGNDATA_{measure_string}_as_f_n_structures_{datastring}_"
                                                      f"{ligand_multiplicity_string}_{normalize_string}.svg"))


