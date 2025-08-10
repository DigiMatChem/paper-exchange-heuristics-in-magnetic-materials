import json
from monty.json import MontyDecoder
import pandas as pd
import plotly.graph_objects as go
from pymatgen.core import Element, Structure
from scipy.stats import kstest

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import (get_mp_magnetic_edge_information,
                                                                               get_bond_angle_occurrences)
from utils_kga.statistical_analysis.ks_test import compute_ks_weighted
from utils_kga.general import pretty_plot


with open("data/unique_4559_cnfeat_MP_db_from_api.json") as f:
    df = json.load(f, cls=MontyDecoder)

ks_dict= {}
string_summary = ""
bond_angle_occurrences = {}
thresholds = [0.05, 0.3, 0.5, 0.8, 1.0]
zero_magmom_failures = {k: [] for k in thresholds}  # due to non-zero magmoms at anionic sites at given threshold
isolated_sites_count = {k: [] for k in thresholds}

for zero_magmom_threshold in thresholds:
    all_stats_dict = {}
    for mp_id, row in df.iterrows():
        structure = row["structure_dict"]
        coordination_features = json.loads(row["coordination_features"], cls=MontyDecoder)
        try:
            edge_result = get_mp_magnetic_edge_information(structure=structure,
                                                           coordination_features=coordination_features,
                                                           magmoms=structure.site_properties["magmom"],
                                                           zero_magmom_threshold=zero_magmom_threshold)
        except AssertionError as e:
            zero_magmom_failures[zero_magmom_threshold].append(
                (str(e), mp_id, structure.as_dict(),structure.site_properties["magmom"]))
            continue

        if not edge_result.empty:  # This can be empty in the case of isolated magnetic sites
            edge_result["ligand_el_set"] = edge_result["ligand_elements"].apply(lambda ls: set(ls))
            edge_result["site_is_tm"] = edge_result["site_element"].apply(lambda el: Element(el).is_transition_metal)
            edge_result["site_to_is_tm"] = edge_result["site_to_element"].apply(lambda el: Element(el).is_transition_metal)
            all_stats_dict[mp_id] = edge_result
        else:
            isolated_sites_count[zero_magmom_threshold].append(mp_id)

    str_sum = f"Zero magmom threshold: {zero_magmom_threshold}, "\
              f"{len(zero_magmom_failures[zero_magmom_threshold])} failed structures, "\
              f"{len(isolated_sites_count[zero_magmom_threshold])} structures with isolated magnetic sites, "\
              f"{len(all_stats_dict)} structures analyzed\n"
    print(str_sum)
    string_summary += str_sum

    # Plot spin bond angle trends of whole db and of KGA-interesting subset of connected TM octahedra
    write_mode = "w"
    for datastring in ["connected_TM_sites",
                       "oxygen_connected_TM_sites",
                       "connected_TM_octahedra",
                       "oxygen_connected_TM_octahedra"]:
        for ligand_multiplicity_bool, ligand_multiplicity_string in zip([True, False],
                                                                        ["ligand_multiplicity_included",
                                                                         "no_ligand multiplicity_included"]):
            for normalize_bool, normalize_string in zip([False, True], ["absolute_occurrences", "normalized_occurrences"]):
                entries = 0
                all_spin_occus = []
                for md_id, ang_df in all_stats_dict.items():
                    test_df = ang_df.loc[(ang_df["site_is_tm"]) & (ang_df["site_to_is_tm"])]
                    if "TM_octahedra" in datastring:
                        test_df = test_df.loc[(test_df["site_ce"] == "O:6") & (test_df["site_to_ce"] == "O:6")]
                    if "oxygen" in datastring:
                        test_df = test_df.loc[test_df["ligand_el_set"]=={"O"}]
                    if not test_df.empty:
                        entries += 1
                        n_lattice_points = df.at[md_id, "n_mag_lattice_points"]
                        occus = get_bond_angle_occurrences(df=test_df,
                                                                         include_ligand_multiplicity=ligand_multiplicity_bool,
                                                                         normalize=normalize_bool,
                                                                         n_lattice_points=n_lattice_points,
                                                                         spin_angle_round=-1,
                                                                         bond_angle_round=7)
                        all_spin_occus.extend(occus)

                all_spin_occus_df = pd.DataFrame(columns=["spin_angle", "bond_angle", "occurrence"], data=all_spin_occus)

                # Create and save one-dimensional histograms of bond angle occurrences as f(spin angle)
                one_d_fig = go.Figure(layout=go.Layout(xaxis=go.layout.XAxis(title="Bond angle (°)"),
                                                       yaxis=go.layout.YAxis(title="Occurrence"),
                                                       title=f"MP: ({entries} structures with connected mag. sites "
                                                             f"({datastring})), "
                                                             f"{ligand_multiplicity_string}, {normalize_string}"))

                for spin_ang in sorted(set(all_spin_occus_df["spin_angle"].values)):
                    sub_df = all_spin_occus_df.loc[all_spin_occus_df["spin_angle"] == spin_ang]

                    bond_angle_occurrences[(f"{zero_magmom_threshold}_zeromagmomthr_"
                                            f"{datastring}_{ligand_multiplicity_string}"
                                            f"_{normalize_string}_{spin_ang}")] = {
                        "fraction_of_bond_angles_over_110deg": sub_df.loc[sub_df["bond_angle"] > 110.0]["occurrence"].values.sum() /
                      sub_df.loc[sub_df["bond_angle"] > 0.0]["occurrence"].values.sum(),
                        "fraction_of_bond_angles_between_75_and_105deg": sub_df.loc[
                                                                             (sub_df["bond_angle"] <= 105.0)
                                                                             & (sub_df["bond_angle"] >= 75.0)]["occurrence"].values.sum() /
                      sub_df.loc[sub_df["bond_angle"] > 0.0]["occurrence"].values.sum()
                    }


                    one_d_fig.add_trace(go.Histogram(
                        histfunc="sum",
                        x=sub_df["bond_angle"].values,
                        y=sub_df["occurrence"].values,
                        nbinsx=181,
                        name=spin_ang
                    ))
                one_d_fig = pretty_plot(one_d_fig)
                one_d_fig.update_layout(legend_title_text="Spin angle (°)")
                one_d_fig.update_layout(title=dict(font=dict(size=10)))
                one_d_fig.update_layout(barmode="overlay")
                one_d_fig.update_traces(opacity=0.75)
                with open(f"data/MP_spin_bond_angle_trends_{zero_magmom_threshold}"
                          f"_magmom_threshold.html", write_mode) as f:
                    f.write(one_d_fig.to_html(full_html=False, include_plotlyjs="cdn"))
                write_mode = "a"
                # Plot special occus to pdf
                if zero_magmom_threshold == 0.5 and ligand_multiplicity_bool == False:
                    for spin_ang in sorted(set(all_spin_occus_df["spin_angle"].values)):
                        one_d_fig = go.Figure(layout=go.Layout(xaxis=go.layout.XAxis(title="Bond angle (°)"),
                                                               yaxis=go.layout.YAxis(title="Occurrence"),
                                                               title=f"MP: ({entries} structures with connected mag. sites "
                                                                     f"({datastring})), "
                                                                     f"{ligand_multiplicity_string}, {normalize_string}"))
                        sub_df = all_spin_occus_df.loc[all_spin_occus_df["spin_angle"] == spin_ang]

                        one_d_fig.add_trace(go.Histogram(
                            histfunc="sum",
                            x=sub_df["bond_angle"].values,
                            y=sub_df["occurrence"].values,
                            nbinsx=181,
                            name=spin_ang
                        ))
                        one_d_fig = pretty_plot(one_d_fig)
                        one_d_fig.update_layout(title=dict(font=dict(size=10)))
                        if not normalize_bool and datastring == "connected_TM_octahedra":
                            one_d_fig.update_layout(xaxis_range=[57, 182])
                        elif normalize_bool and datastring == "connected_TM_octahedra":
                            one_d_fig.update_layout(xaxis_range=[57, 182])
                        one_d_fig.write_image(f"data/MP_spin_bond_angle_trends_{zero_magmom_threshold}_"
                                              f"magmom_threshold_{datastring}_{ligand_multiplicity_string}_"
                                              f"{normalize_string}_{spin_ang}_spin_ang.pdf")

                # Compute KS Test
                fm_occus = [ls for ls in all_spin_occus if ls[0] == 0.0]
                afm_occus = [ls for ls in all_spin_occus if ls[0] == 180.0]
                assert len(fm_occus) + len(afm_occus) == len(all_spin_occus)

                weighted_ks_test_d, weighted_ks_test_p = compute_ks_weighted(afm_occus, fm_occus)

                # Sanity check: compute ks test normally if absolute occurrences and compare (ligand mult!!)
                if set([ls[2] % 1 == 0 for ls in all_spin_occus]) == {True}:
                    # Revert back to single occurrences of AFM and FM subset
                    fm_occus_flatten = []
                    for ls in fm_occus:
                        fm_occus_flatten.extend([ls[1] for _ in range(int(ls[2]))])
                    afm_occus_flatten = []
                    for ls in afm_occus:
                        afm_occus_flatten.extend([ls[1] for _ in range(int(ls[2]))])
                    scipy_ks_test = kstest(afm_occus_flatten, fm_occus_flatten, alternative="two-sided", method="exact")
                    assert round(getattr(scipy_ks_test, "statistic"), 10) == abs(round(weighted_ks_test_d, 10))
                    assert round(getattr(scipy_ks_test, "pvalue"), 16) == round(weighted_ks_test_p, 16)

                ks_dict[f"{zero_magmom_threshold}_zeromagmomthr_{datastring}_{ligand_multiplicity_string}_{normalize_string}"] = {
                    "ks_test_p_value": weighted_ks_test_p,
                    "ks_test_statistic": weighted_ks_test_d,
                    "n_afm": round(sum([l[2] for l in afm_occus]), 2),
                    "n_fm": round(sum([l[2] for l in fm_occus]), 2),
                }

with open("data/zero-magmom-failures.json", "w") as f:
    json.dump(zero_magmom_failures, f)
with open("data/ks_test_results.json", "w") as f:
    json.dump(ks_dict, f)
with open("data/bond_angle_occurrences.json", "w") as f:
    json.dump(bond_angle_occurrences, f)
with open("data/string_summary.txt", "w") as f:
    f.write(string_summary)


