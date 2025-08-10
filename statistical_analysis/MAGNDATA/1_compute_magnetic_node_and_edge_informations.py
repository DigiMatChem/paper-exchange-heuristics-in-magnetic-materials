import json
from monty.json import MontyDecoder

from utils_kga.statistical_analysis.get_spin_and_bond_angle_statistics import (get_magnetic_edge_information,
                                                                               get_magnetic_node_information)


all_stats_dict = {"magnetic_node_information": {},
                  "magnetic_edge_information": {}
                  }

with open("../../data_retrieval_and_preprocessing_MAGNDATA/data/df_grouped_and_chosen_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)


for row_id, row in df.iterrows():
    if row["chosen_one"]:
        structure = json.loads(row["mag_structure"], cls=MontyDecoder)
        coordination_features = json.loads(row["coordination_features"], cls=MontyDecoder)

        node_result = get_magnetic_node_information(structure=structure,
                                                    coordination_features=coordination_features,
                                                    spin_angle_tols=(0.0, 10.0, 20.0, 50.0, 80.0))
        if not node_result.empty:
            all_stats_dict["magnetic_node_information"][row_id] = node_result
        else:  # This should not be the case!
            raise ValueError(f"Structure without magnetic sites! ({row_id})")

        edge_result = get_magnetic_edge_information(structure=structure,
                                                    coordination_features=coordination_features,
                                                    only_include_pure_oxygen_edges=False,
                                                    spin_angle_tols=(0.0, 10.0, 20.0, 50.0, 80.0))
        if not edge_result.empty:  # This can be empty in the case of isolated magnetic sites
            all_stats_dict["magnetic_edge_information"][row_id] = edge_result

for description, df_dict in all_stats_dict.items():
    with open(f"data/dfs_of_{description}.json", "w") as f:
        json.dump({key: df.to_dict() for key, df in df_dict.items()}, f)
