"""
Tests concerning the spin and bond angle trend analyses of MP.
"""

import json
import numpy as np
import pytest
from monty.json import MontyDecoder
from pymatgen.core import Element
from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import (
    get_mp_magnetic_edge_information,
    get_bond_angle_occurrences,
)


def test_large_bond_angle_magnetism_occurrences(mp_db_dir):
    """
    Count occurences in stable and unique MP database that have bond angles between
    magnetically ordered TM octahedra that are AFM / FM connections and test that they are in
    similar range as expected from statistical analysis.
    """
    with open(mp_db_dir) as f:
        df = json.load(f, cls=MontyDecoder)
    zero_magmom_threshold = 0.5
    afm, fm = [], []

    for row_id, row in df.iterrows():
        structure = json.loads(row["structure"], cls=MontyDecoder)
        coordination_features = json.loads(row["coordination_features"], cls=MontyDecoder)
        try:
            edge_result = get_mp_magnetic_edge_information(
                structure=structure,
                coordination_features=coordination_features,
                magmoms=row["magmoms"]["vasp"],
                zero_magmom_threshold=zero_magmom_threshold,
            )
        except AssertionError as e:
            continue

        if not edge_result.empty:  # This can be empty in the case of isolated magnetic sites
            edge_result["ligand_el_set"] = edge_result["ligand_elements"].apply(lambda ls: set(ls))
            edge_result["site_is_tm"] = edge_result["site_element"].apply(lambda el: Element(el).is_transition_metal)
            edge_result["site_to_is_tm"] = edge_result["site_to_element"].apply(
                lambda el: Element(el).is_transition_metal
            )
            edge_result["any_ang"] = edge_result["mag_ligand_mag_angle"].apply(
                lambda l: any([True for b in l if 110 < b <= 180])
            )  # okay approach here as corner connections at higher angles between octahedra

            test_df = edge_result.loc[(edge_result["site_is_tm"]) & (edge_result["site_to_is_tm"])]
            test_df = test_df.loc[
                (test_df["site_ce"] == "O:6") & (test_df["site_to_ce"] == "O:6") & (test_df["any_ang"])
            ]
            afm.append(len(test_df.loc[test_df["spin_angle"] == 180]) / len(edge_result))
            fm.append(len(test_df.loc[test_df["spin_angle"] == 0]) / len(edge_result))

    # Assert that similar values
    print(sum(afm), sum(fm))
    assert abs(sum(afm) - sum(fm)) < sum(fm) * 0.05


@pytest.mark.parametrize(
    "index_nr, include_ligand_multiplicity, normalize, expected_spin_angles, expected_bond_angles, expected_occurrences",
    [
        (
            "28",
            False,
            False,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [130.0372971, 130.1332479, 130.1331075, 130.037169, 99.8267864, 99.8270521],
            [4.0, 4.0, 4.0, 4.0, 2.0, 2.0],
        ),
        (
            "46",
            False,
            False,
            [0.0, 0.0, 0.0, 0.0, 180.0, 180.0, 0.0, 0.0, 180.0, 180.0, 180.0, 180.0],
            [
                135.2806721,
                135.2801808,
                135.2960142,
                135.2951645,
                135.1267992,
                135.1274486,
                135.2880891,
                135.2879268,
                135.1373368,
                135.1375105,
                135.1206164,
                135.1217419,
            ],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        (
            "3103",
            True,
            False,
            [0.0, 0.0, 180.0, 180.0, 180.0, 180.0, 0.0, 0.0],
            [95.1448994, 95.1425107, 94.3677872, 94.3677888, 94.4105561, 94.4086808, 95.0858881, 95.0886528],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        (
            "9522",
            False,
            True,
            [0.0, 0.0, 0.0, 0.0],
            [132.8614595, 132.8595786, 132.851274, 132.8505533],
            [0.25, 0.25, 0.25, 0.25],
        ),
    ],
)
def test_bond_angle_occurrences_mp_db(
    mp_db_dir,
    index_nr,
    include_ligand_multiplicity,
    normalize,
    expected_spin_angles,
    expected_bond_angles,
    expected_occurrences,
):
    with open(mp_db_dir) as f:
        df = json.load(f, cls=MontyDecoder)

    structure = MontyDecoder().decode(df.loc[index_nr, "structure"])
    coordination_features = MontyDecoder().decode(df.loc[index_nr, "coordination_features"])

    edge_result = get_mp_magnetic_edge_information(
        structure=structure,
        coordination_features=coordination_features,
        magmoms=df.loc[index_nr, "magmoms"]["vasp"],
        zero_magmom_threshold=0.5,
    )

    edge_result["ligand_el_set"] = edge_result["ligand_elements"].apply(lambda ls: set(ls))
    edge_result["site_is_tm"] = edge_result["site_element"].apply(lambda el: Element(el).is_transition_metal)
    edge_result["site_to_is_tm"] = edge_result["site_to_element"].apply(lambda el: Element(el).is_transition_metal)

    bond_angle_occurrences = np.array(
        get_bond_angle_occurrences(
            df=edge_result,
            include_ligand_multiplicity=include_ligand_multiplicity,
            normalize=normalize,
            n_lattice_points=df.at[index_nr, "n_mag_lattice_points"],
            spin_angle_round=-1,
            bond_angle_round=7,
        )
    )

    assert bond_angle_occurrences[:, 0].tolist() == pytest.approx(expected_spin_angles, rel=1e-5, abs=1e-8)
    assert bond_angle_occurrences[:, 1].tolist() == pytest.approx(expected_bond_angles, rel=1e-5, abs=1e-8)
    assert bond_angle_occurrences[:, 2].tolist() == pytest.approx(expected_occurrences, rel=1e-5, abs=1e-8)
