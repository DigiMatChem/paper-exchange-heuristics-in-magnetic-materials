"""
Tests concerning the spin and bond angle trend analyses of MAGNDATA.
"""

import json

from monty.json import MontyDecoder
import os
from pymatgen.core import Element
from pymatgen.io.cif import CifParser

from utils_kga.general import (
    get_coordination_features_and_supercell_coordination_features,
)
from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import *


def test_general(mcif_dir):
    """
    General sanity checks for trend analyses - as in p, ap score tests, coordination features is tested if scales with
    supercell and normalization of statistics utils functions is tested.
    """
    # For a given structure, assert that the relative occurrences of
    # spin and bond angles is the same for its 2x2x2 supercell
    for md_id in ["0.15", "0.16"]:
        struct = CifParser(os.path.join(mcif_dir, md_id + ".mcif")).parse_structures(primitive=False)[0]
        super_struct, cn_feat, super_cn_feat = get_coordination_features_and_supercell_coordination_features(
            structure=struct
        )
        df = get_magnetic_edge_information(structure=struct, coordination_features=cn_feat)
        super_df = get_magnetic_edge_information(structure=super_struct, coordination_features=super_cn_feat)

        assert set([tuple(o) for o in get_bond_angle_occurrences(df=df, normalize=True)]) == set(
            [tuple(o) for o in get_bond_angle_occurrences(df=super_df, normalize=True)]
        )


def test_oct_oct_counts():
    """
    Sanity check: asserts number of structures that contribute to almost collinear interactions
    between octahedrally coordinated TM sites / RE sites.
    Tests by calculating occurrences from scratch, not with approach of this submodule.
    """
    with open(
        "data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json"
    ) as f:
        df = json.load(f, cls=MontyDecoder)
    tm_tm_structs, re_re_structs = set(), set()
    for row_id, row in df.iterrows():
        if row["chosen_one"]:
            coordination_features = json.loads(row["coordination_features"], cls=MontyDecoder)
            structure = json.loads(row["mag_structure"], cls=MontyDecoder)
            site_ce_dict = dict(
                zip(
                    [i["site"] for i in coordination_features.ces],
                    [i["ce_symbols"][0] for i in coordination_features.ces],
                )
            )
            if "O:6" in list(site_ce_dict.values()):
                for n in coordination_features.ce_neighbors:
                    if (
                        site_ce_dict[n["site"]] == "O:6"
                        and site_ce_dict[n["site_to"]] == "O:6"
                        and any(structure.sites[n["site"]].properties["magmom"].get_moment())
                        and any(structure.sites[n["site_to"]].properties["magmom"].get_moment())
                    ):
                        tm_1 = Element(coordination_features.sites.elements[n["site"]]).is_transition_metal
                        tm_2 = Element(coordination_features.sites.elements[n["site_to"]]).is_transition_metal
                        re_1 = Element(coordination_features.sites.elements[n["site"]]).is_rare_earth_metal
                        re_2 = Element(coordination_features.sites.elements[n["site_to"]]).is_rare_earth_metal
                        if tm_1 and tm_2:
                            tm_tm_structs.add(row_id)
                        elif re_1 and re_2:
                            re_re_structs.add(row_id)

    assert len(tm_tm_structs) == 394
    assert len(re_re_structs) == 41


def test_tm_octahedra_subset_analysis():
    """Test analysis of node property (csm) as a function of certain bond- and spin angle occurrences."""
    results = [[True, {1.09: 16, 3.28: 24}], [False, {1.09: 5.33, 3.28: 9.33}]]
    with open("statistical_analysis/MAGNDATA/data/dfs_of_magnetic_edge_information.json") as f:
        dict_all_stats = json.load(f)
    test = pd.DataFrame.from_dict(dict_all_stats["0.260"])
    test["site_is_tm"] = test["site_element"].apply(lambda el: Element(el).is_transition_metal)
    test["site_to_is_tm"] = test["site_to_element"].apply(lambda el: Element(el).is_transition_metal)
    subdf = test.loc[
        (test["site_ce"] == "O:6")
        & (test["site_to_ce"] == "O:6")
        & (test["site_is_tm"])
        & (test["site_to_is_tm"])
        & (test["spin_angle"] <= 1.0)
    ]
    for ligand_multiplicity_bool, result_dict in results:
        occus = get_bond_angle_interval_statistics(
            df=subdf,
            include_ligand_multiplicity=ligand_multiplicity_bool,
            analyze_column="site_csm",
            n_lattice_points=1,
            bond_angle_interval=(85, 95),
        )
        for csm, occu in result_dict.items():
            assert occu == round(sum(li[1] for li in occus if li[0] == csm), 2)
