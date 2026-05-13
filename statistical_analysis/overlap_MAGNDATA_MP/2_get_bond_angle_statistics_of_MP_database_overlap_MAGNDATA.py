import json
from monty.json import MontyDecoder
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from pymatgen.core import Element

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import get_mp_magnetic_edge_information, get_bond_angle_occurrences
from utils_kga.general import pretty_plot



def binned_distribution(df, x_value, bins):
    subset = df[df["spin_angle"] == x_value]
    hist, _ = np.histogram(
        subset["bond_angle"],
        bins=bins,
        weights=subset["occurrence"]
    )
    return hist



overlap_data_dir = "data"
zero_magmom_threshold = 0.5


# Match only collinear entries of cryst. unique MAGNDATA
match_df = pd.read_json(os.path.join(overlap_data_dir, "matches_mp_magndata.json"))
match_df = match_df.loc[(match_df["magndata_is_collinear"])
                        & (match_df["magndata_is_chosen_one"])]

matches_mp_part = [str(v) for v in match_df["mp_id"].values]
matches_md_part = match_df["magndata_id"].values


# MAGNDATA
# Load edge-df
with open("../MAGNDATA/data/dfs_of_magnetic_edge_information.json") as f:
    dict_all_stats_md = json.load(f)
all_stats_md = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats_md.items()}

# Add is_tm bool for later easier analysis
for ang_df in all_stats_md.values():
    ang_df["site_is_tm"] = ang_df["site_element"].apply(lambda el: Element(el).is_transition_metal)
    ang_df["site_to_is_tm"] = ang_df["site_to_element"].apply(lambda el: Element(el).is_transition_metal)
    ang_df["site_ion"] = ang_df.apply(lambda r: r["site_element"] + str(r["site_oxidation"]), axis=1)
    ang_df["site_to_ion"] = ang_df.apply(lambda r: r["site_to_element"] + str(r["site_to_oxidation"]), axis=1)
    ang_df["ligand_el_set"] = ang_df["ligand_elements"].apply(lambda ls: set(ls))

# For metadata filtering and computation of occurrences in magnetic primitive cells
with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df_md = json.load(f, cls=MontyDecoder)

# MP
with open("../MP/data_and_plots/df_stable_and_unique_MP_db.json") as f:
    df_mp = json.load(f, cls=MontyDecoder)

# MP imports
with open(os.path.join("../MP/data_and_plots/", "dfs_of_magnetic_edge_information_0p5_zero-magmom_threshold.json")) as f:
    dict_all_stats = json.load(f)
all_stats_dict_mp = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats.items()}


# Compute difference MP-MAGNDATA f. TM octahedra of same ion
ligand_multiplicity_bool = False
ligand_multiplicity_string = "no_ligand_multiplicity_included"
datastring = "connected_TM_octahedra_same_ions"

for normalize_bool, normalize_string in zip([False, True], ["absolute_occurrences", "normalized_occurrences"]):

    # Count MP occurrences
    entries_mp = 0
    all_spin_occus_mp = []

    for mp_id, ang_df_mp in all_stats_dict_mp.items():
        if mp_id in matches_mp_part:

            test_df = ang_df_mp.loc[(ang_df_mp["site_is_tm"]) & (ang_df_mp["site_to_is_tm"])]
            test_df = test_df.loc[(test_df["site_ce"] == "O:6") & (test_df["site_to_ce"] == "O:6")]
            test_df = test_df.loc[(test_df["site_element"] == test_df["site_to_element"])
                                    & (test_df["site_oxidation"] == test_df["site_to_oxidation"])]
                
            if not test_df.empty:
                entries_mp += 1
                n_lattice_points = df_mp.at[mp_id, "n_mag_lattice_points"]
                occus_mp = get_bond_angle_occurrences(df=test_df,
                                                                    include_ligand_multiplicity=ligand_multiplicity_bool,
                                                                    normalize=normalize_bool,
                                                                    n_lattice_points=n_lattice_points,
                                                                    spin_angle_round=-1,
                                                                    bond_angle_round=7)
                all_spin_occus_mp.extend(occus_mp)

    all_spin_occus_df_mp = pd.DataFrame(columns=["spin_angle", "bond_angle", "occurrence"], data=all_spin_occus_mp)

    # Repeat for magndata
    entries_md = 0
    all_spin_occus_md = []

    for md_id, ang_df in all_stats_md.items():
        if md_id in matches_md_part:

            test_df = ang_df.loc[(ang_df["site_is_tm"]) & (ang_df["site_to_is_tm"])]
            test_df = test_df.loc[(test_df["site_ce"] == "O:6") & (test_df["site_to_ce"] == "O:6")]
            test_df = test_df.loc[(test_df["site_element"] == test_df["site_to_element"])
                                    & (test_df["site_oxidation"] == test_df["site_to_oxidation"])]
            
            if not test_df.empty:

                entries_md += 1
                n_lattice_points = df_md.at[md_id, "n_lattice_points"]
                occus_md = get_bond_angle_occurrences(df=test_df,
                                                                include_ligand_multiplicity=ligand_multiplicity_bool,
                                                                normalize=normalize_bool,
                                                                n_lattice_points=n_lattice_points,
                                                                spin_angle_round=-1,
                                                                bond_angle_round=7)

                all_spin_occus_md.extend(occus_md)
    all_spin_occus_df_md = pd.DataFrame(columns=["spin_angle", "bond_angle", "occurrence"], data=all_spin_occus_md)
    
    assert set(all_spin_occus_df_md["spin_angle"].values) == {0.0, 180.0}  # Sanity check that no intermediate spin angles (specifically 5-10 deg)
    print(entries_md, entries_mp)

    # Bin into same bins and plot bond angle histograms and difference between histograms
    nbins = 80

    y_min = min(all_spin_occus_df_md["bond_angle"].min(), all_spin_occus_df_mp["bond_angle"].min())
    y_max = max(all_spin_occus_df_md["bond_angle"].max(), all_spin_occus_df_mp["bond_angle"].max())

    bins = np.linspace(y_min, y_max, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # MP and MAGNDATA histograms
    dist_mp = {x: binned_distribution(all_spin_occus_df_mp, x, bins) for x in [0, 180]}
    dist_md = {x: binned_distribution(all_spin_occus_df_md, x, bins) for x in [0, 180]}

    # Difference histogram
    diff = {x: dist_mp[x] - dist_md[x] for x in [0, 180]}

    # Have a look at FM distributions
    spin_ang = 0
    
    for d, d_name in zip([dist_mp, dist_md], ["MP", "MAGNDATA"]):

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=bin_centers,
            y=d[spin_ang],
            marker_color="#025268"            
        ))
        fig = pretty_plot(fig)

        fig.update_layout(
            title=dict(text=f"Binned bond angle distribution ({d_name} {spin_ang}° spin angle) {datastring} {ligand_multiplicity_string} {normalize_string}", font=dict(size=9)),
            xaxis_title="Bond angle",
            yaxis_title="Occurrence",            
        )
        

        if normalize_bool:
            fig.update_layout(yaxis_range=[0, 4.2], xaxis_range=[62, 182])
        else:
            fig.update_layout(yaxis_range=[0, 86], xaxis_range=[62, 182])
        
        fig.update_layout(
                margin=dict(
                    l=80,
                    r=80,
                    t=140,
                    b=80
                )
            )

        fig.write_image(os.path.join(overlap_data_dir, 
                                     f"bond_angle_distribution_{d_name}_{spin_ang}_spinang_{datastring}_{ligand_multiplicity_string}_{normalize_string}.pdf"))


    # Plot difference between histograms
    fig_diff = go.Figure()

    fig_diff.add_trace(go.Bar(
        x=bin_centers,
        y=diff[spin_ang],
        marker_color="#025268"
    ))
    fig_diff = pretty_plot(fig_diff)

    fig_diff.update_layout(
        title=dict(text=f"Bond angle distribution difference (MP - MAGNDATA),{spin_ang}° spin angle) {datastring} {ligand_multiplicity_string} {normalize_string}", font=dict(size=9)),
        xaxis_title="Bond angle",
        yaxis_title="Occurrence difference",
    )
    fig_diff.update_layout(xaxis_range=[62, 182])
    fig_diff.update_layout(
                margin=dict(
                    l=80,
                    r=80,
                    t=160,
                    b=80
                )
            )
    
    fig_diff.write_image(os.path.join(overlap_data_dir, 
                                 f"difference_bond_angle_distribution_MP-MAGNDATA_{spin_ang}_spinang_{datastring}_{ligand_multiplicity_string}_{normalize_string}.pdf"))

