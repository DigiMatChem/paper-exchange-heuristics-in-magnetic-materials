"""Do analysis of edges between magnetic nodes for all structures before multiples elimination to demonstrate
effects are not due to chosen set of crystallographic uniques / due to multiples elimination method."""
import json
from monty.json import MontyDecoder

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import get_magnetic_edge_information


edge_dict = {}

# df contains info on all magnetic structures, also those not chosen
with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)

for row_id, row in df.iterrows():
    structure = json.loads(row["mag_structure"], cls=MontyDecoder)
    coordination_features = json.loads(row["coordination_features"], cls=MontyDecoder)

    edge_result = get_magnetic_edge_information(structure=structure,
                                                coordination_features=coordination_features,
                                                only_include_pure_oxygen_edges=False,
                                                spin_angle_tols=(0.0, 10.0, 20.0, 50.0, 80.0))
    if not edge_result.empty:  # This can be empty in the case of isolated magnetic sites
        edge_dict[row_id] = edge_result

with open(f"data/dfs_of_magnetic_edge_information_include_crystallographic_multiples.json", "w") as f:
    json.dump({key: df.to_dict() for key, df in edge_dict.items()}, f)
