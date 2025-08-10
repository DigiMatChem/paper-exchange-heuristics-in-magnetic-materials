"""Tests for computing the p and ap score
(supercell score testing for whole db see test_supercell_p_ap_score.py, this script also
tests for specific values)."""
from pymatgen.io.cif import CifParser
import os
import pytest

from utils_kga.data_retrieval_and_preprocessing.get_p_ap_scores import get_p_ap_scores
from utils_kga.general import get_coordination_features_and_supercell_coordination_features


def test_get_p_ap_scores(mcif_dir):
    scores_dict_general = {
        "0.17": (0.0, 1.0),
        "0.44": (0.0, 0.0),
        "0.191": (0.5, 0.5)
    }

    for md_id, expected_scores in scores_dict_general.items():
        struct = CifParser(os.path.join(mcif_dir, md_id + ".mcif")).parse_structures(primitive=False)[0]
        super_struct, cn_feat, super_cn_feat = get_coordination_features_and_supercell_coordination_features(
            structure=struct)

        scores = get_p_ap_scores(structure=struct, coordination_features=cn_feat)
        super_scores = get_p_ap_scores(structure=super_struct, coordination_features=super_cn_feat)

        assert scores == expected_scores
        assert scores == super_scores

    scores_dict_spin_ligand_mult = {"0.16": ((0.666667, 0.333333), (0.333333, 0.666667))}

    for md_id, expected_scores in scores_dict_spin_ligand_mult.items():
        struct = CifParser(os.path.join(mcif_dir, md_id + ".mcif")).parse_structures(primitive=False)[0]
        super_struct, cn_feat, super_cn_feat = get_coordination_features_and_supercell_coordination_features(
            structure=struct)

        scores0 = get_p_ap_scores(structure=struct,
                                  coordination_features=cn_feat,
                                  weighting="exclude_ligand_multiplicities")

        scores1 = get_p_ap_scores(structure=super_struct,
                                  coordination_features=super_cn_feat,
                                  weighting="include_ligand_multiplicities")

        super_scores0 = get_p_ap_scores(structure=super_struct,
                                        coordination_features=super_cn_feat,
                                        weighting="exclude_ligand_multiplicities")

        super_scores1 = get_p_ap_scores(structure=struct,
                                        coordination_features=cn_feat,
                                        weighting="include_ligand_multiplicities")

        assert scores0 == super_scores0
        assert scores1 == super_scores1

        assert scores0 == pytest.approx(expected_scores[0], abs=1e-05)
        assert scores1 == pytest.approx(expected_scores[1], abs=1e-05)

