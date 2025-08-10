"""
Test featurization outcome of own (pycoordinationnet-based) functionality.
"""
from monty.serialization import loadfn
import os
from pymatgen.core import Species
from pymatgen.io.cif import CifParser

from utils_kga.featurization.structurewise_coordinationnet_features import *
from utils_kga.general import get_coordination_features_and_supercell_coordination_features


def test_magnetic_site_guessing_methods(mcif_dir):
    """Assert that expected number of sites are featurized with different guessing methods for magnetic sites."""
    structure = CifParser(os.path.join(mcif_dir, "0.1.mcif")).parse_structures(primitive=False)[0]
    magmom_info = tuple(loadfn("featurization/default_magmoms_uncommented.yaml").keys())
    cn_feat = CoordinationFeatures().from_structure(structure=structure,
                                                    env_strategy="simple",
                                                    guess_oxidation_states_from_composition=False,
                                                    include_edge_multiplicities=True)
    feat = get_structurewise_coordinationnet_features(
            structure=structure,
            coordination_features=cn_feat,
            site_filters=(d_f_cation_filter, magnetic_cation_filter),
            stats=("mean", "min"),
            magmom_info=magmom_info
        )
    la3_rad = Species("La3+").ionic_radius
    mn3_rad = Species("Mn3+").ionic_radius

    assert feat["cn_species_ionic_radius_mean_dfcat"] == (la3_rad + mn3_rad) / 2
    assert feat["cn_species_ionic_radius_mean_magcat"] == mn3_rad


def test_pycoordinationnet_features(mcif_dir):
    """
    Test featurization outcome of pycoordinationnet features.
    """
    structure = CifParser(os.path.join(mcif_dir, "0.1.mcif")).parse_structures(primitive=False)[0]
    magmom_info = tuple(loadfn("featurization/default_magmoms_uncommented.yaml").keys())
    super_structure, cn_feat, super_cn_feat = get_coordination_features_and_supercell_coordination_features(
        structure=structure, supercell_matrix=2)
    feat = get_structurewise_coordinationnet_features(
        structure=structure,
        coordination_features=cn_feat,
        site_filters=(d_f_cation_filter, magnetic_cation_filter),
        stats=("mean", "min"),
        magmom_info=magmom_info
    )
    super_feat = get_structurewise_coordinationnet_features(
        structure=super_structure,
        coordination_features=super_cn_feat,
        site_filters=(d_f_cation_filter, magnetic_cation_filter),
        stats=("mean", "min"),
        magmom_info=magmom_info
    )

    assert round(feat["cn_corner_perc_dfcat"] + feat["cn_edge_perc_dfcat"] + feat["cn_face_perc_dfcat"], 6) == 1.0
    assert round(feat["cn_corner_perc_magcat"] + feat["cn_edge_perc_magcat"] + feat["cn_face_perc_magcat"], 6) == 1.0

    assert round(feat["cn_corner_perc_magcat"], 6) == 1.0

    for f, f_val in feat.items():
        assert round(f_val, 6) == round(super_feat[f], 6)
