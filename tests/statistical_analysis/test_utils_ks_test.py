import json
from monty.json import MontyDecoder
from pymatgen.core import Element
from scipy.stats import kstest

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import *
from utils_kga.statistical_analysis.ks_test import *


def test_own_ks_test():
    # Load edge-df
    with open("statistical_analysis/MAGNDATA/data/dfs_of_magnetic_edge_information.json") as f:
        dict_all_stats = json.load(f)
    all_stats = {key: pd.DataFrame.from_dict(df) for key, df in dict_all_stats.items()}

    # For metadata filtering
    with open(
        "data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json"
    ) as f:
        df = json.load(f, cls=MontyDecoder)

    # Add is_tm bool for later easier analysis
    for ang_df in all_stats.values():
        ang_df["site_is_tm"] = ang_df["site_element"].apply(lambda el: Element(el).is_transition_metal)
        ang_df["site_to_is_tm"] = ang_df["site_to_element"].apply(lambda el: Element(el).is_transition_metal)

    all_spin_occus = []
    for md_id, ang_df in all_stats.items():
        subdf = ang_df.loc[
            (ang_df["site_ce"] == "O:6")
            & (ang_df["site_to_ce"] == "O:6")
            & (ang_df["site_is_tm"])
            & (ang_df["site_to_is_tm"])
        ]
        if not subdf.empty:
            n_lattice_points = df.at[md_id, "n_lattice_points"]
            occus = get_bond_angle_occurrences(
                df=subdf,
                include_ligand_multiplicity=True,
                normalize=False,
                n_lattice_points=n_lattice_points,
                spin_angle_round=-1,
                bond_angle_round=7,
            )
            all_spin_occus.extend(occus)

    fm_occus = [ls for ls in all_spin_occus if ls[0] == 0.0]
    afm_occus = [ls for ls in all_spin_occus if ls[0] == 180.0]

    fm_occus_flatten = []
    for ls in fm_occus:
        fm_occus_flatten.extend([ls[1] for _ in range(int(ls[2]))])
    afm_occus_flatten = []
    for ls in afm_occus:
        afm_occus_flatten.extend([ls[1] for _ in range(int(ls[2]))])
    scipy_ks_test = kstest(fm_occus_flatten, afm_occus_flatten, alternative="two-sided", method="exact")

    own_ks_test_d, own_ks_test_p = compute_ks_weighted(afm_occus, fm_occus)

    assert round(getattr(scipy_ks_test, "pvalue"), 16) == round(own_ks_test_p, 16)
    assert round(getattr(scipy_ks_test, "statistic"), 10) == abs(round(own_ks_test_d, 10))
