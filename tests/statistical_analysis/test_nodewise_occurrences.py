"""
Test analysis of nodewise occurrences of coordination environments as a function of sitewise collinearity.
"""
import os
from pymatgen.io.cif import CifParser

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import *


def test_ce_collinearity_occurrences(mcif_dir):
    """
    Test correct computation of absolute and normalized occurrences of coordination environment - sitewise collinearity
    occurrences for a small subset of collinear and non-collinear structures.
    """
    coll = "site_collinearity_0.0_exclude_ligand_multiplicities"
    occus = []
    for md_id in ["0.15", "0.220", "0.992"]:
        struct = CifParser(os.path.join(mcif_dir, md_id + ".mcif")).parse_structures(primitive=False)[0]
        cn_feat = CoordinationFeatures().from_structure(
            structure=struct,
            env_strategy="simple",
            guess_oxidation_states_from_composition=False,
            include_edge_multiplicities=True,
        )
        df = get_magnetic_node_information(structure=struct, coordination_features=cn_feat, spin_angle_tols=(0.0, 50.0))

        occus.extend(
            get_node_feature_occurrences(
                df=df,
                normalize=False,
                col0="site_ce",
                col1=coll,
                n_lattice_points=1,  # Count standard representation for test
            )
        )
    occus = pd.DataFrame(columns=["site_ce", coll, "occurrence"], data=occus)

    for condition, result in zip([occus[coll] == 1.0, occus[coll] < 1.0], [{"O:6": 2}, {"SBT:8": 4, "O:6": 12}]):
        sub_df = occus.loc[condition]
        summed_occus = {}
        for x_val in set(sub_df["site_ce"].values):
            summed_occus[x_val] = np.sum(sub_df.loc[sub_df["site_ce"] == x_val]["occurrence"].values)
        assert summed_occus == result
