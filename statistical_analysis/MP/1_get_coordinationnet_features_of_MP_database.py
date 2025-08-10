"""Redo spin-bond-angle analysis for DFT dataset of NC Frey et al. Science Advances 2020, 6 (50), eabd1076.
(data available at https://github.com/ncfrey/magnetic-topological-materials/tree/master).
First part: get coordinationfeatures objects and metadata of stable + unique magnetic structures."""
import json
from monty.serialization import loadfn
from monty.json import MontyEncoder
import os
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from spglib import get_magnetic_symmetry_dataset

from utils_kga.coordination_features import CoordinationFeatures


dump = loadfn("raw_data/magnetic-orderings-database-snapshot.part1.json.gz")
dump.extend(loadfn("raw_data/magnetic-orderings-database-snapshot.part2.json.gz"))

os.makedirs("data_and_plots", exist_ok=True)

row_lists, erroneous = [], []

for entry_id, entry in enumerate(dump):
    if entry["stable"]:
        assert round(entry["energy_above_ground_state_per_atom"], 6) == 0.0
        assert entry["decomposes_to"] is None

        # Attempt coordinationnet featurization
        try:
            cn_feat = CoordinationFeatures().from_structure(structure=entry["structure"],
                                                            env_strategy="simple",
                                                            guess_oxidation_states_from_composition=False,
                                                            include_edge_multiplicities=True)
            oxidation_states_origin = "BVA"
        except (ValueError, TypeError):
            try:
                cn_feat = CoordinationFeatures().from_structure(structure=entry["structure"],
                                                                env_strategy="simple",
                                                                guess_oxidation_states_from_composition=True,
                                                                include_edge_multiplicities=True)
                oxidation_states_origin = "pymatgen_composition_guess"
            except (ValueError, AssertionError) as e:
                erroneous.append((entry_id, entry["task_id"], str(e)))
                continue

        assert entry["structure"].num_sites / entry["parent_structure"].num_sites % 1 == 0, \
            str(entry_id) +  str(entry["structure"].num_sites / entry["parent_structure"].num_sites)

        # Compute primitive magnetic structure and its space group type to arrive at n_lattice points of
        sga_mag = SpacegroupAnalyzer(structure=entry["structure"],
                                                    symprec=1e-5,
                                                    angle_tolerance=-1.0,
                                                    )
        mag_symm_dataset = get_magnetic_symmetry_dataset(cell=sga_mag._cell,
                                                         symprec=1e-5,
                                                         angle_tolerance=-1.0,
                                                         mag_symprec=-1.0,
                                                         )
        prim_mag_lattice_volume = Lattice(mag_symm_dataset["primitive_lattice"]).volume
        n_mag_lattice_points = entry["structure"].volume / prim_mag_lattice_volume

        # Assert that integer number of magnetic lattice points
        assert round(n_mag_lattice_points, 2) % 1 == 0.0, (
                str(entry_id) + ": " + str(entry["structure"].volume + " " +str(prim_mag_lattice_volume)))

        row_lists.append({
            "entry_id": entry_id,
            "task_id": entry["task_id"],
            "formula": entry["formula"],
            "structure": entry["structure"],
            "parent_structure": entry["parent_structure"],
            "coordination_features": cn_feat,
            "oxidation_states_origin": oxidation_states_origin,
            "ordering": entry["ordering"],
            "space_group_type_non_mag": entry["symmetry"],
            "n_mag_lattice_points": round(n_mag_lattice_points, 0),
            "magmoms": entry["magmoms"]
        })

df = pd.DataFrame.from_records(row_lists)
df.set_index("entry_id", inplace=True, drop=True)

df.drop_duplicates(subset=["parent_structure"], inplace=True)
df.drop_duplicates(subset=["formula", "space_group_type_non_mag"], inplace=True)
df = df.loc[df.ordering != "NM"]
# Stronger filtering as only searching for stable entries leaves more structures
# than described in paper by Frey et al., even when eliminating duplicates with the same parent structure.
# To prevent data leakage in the ML part, we filter for unique formula - space group type pairs.
# Also, we exclude StructureMatcher duplicates as is done in the MAGNDATA database.
df["group_index"] = None
max_group_index = 0
s = StructureMatcher()
for idx, (row_id, row) in enumerate(df.iterrows()):
    # Assign multiples group index
    unique = True
    for row_id_comp, row_comp in df[:idx].iterrows():
        if s.fit(row_comp["structure"], row["structure"]):
            unique = False
            df.at[row_id, "group_index"] = row_comp["group_index"]
            break

    if unique:
        max_group_index += 1
        df.at[row_id, "group_index"] = max_group_index

print(len(df))
df.drop_duplicates(subset="group_index", inplace=True)
print(len(df))
df.drop(columns=["group_index"], inplace=True)

with open("data_and_plots/df_stable_and_unique_MP_db.json", "w") as f:
    json.dump(df, f, cls=MontyEncoder)

# Save log of erroneous structures
with open("data_and_plots/erroneous_structures_log_get_coordinationnet_features_of_MP_database.json", "w") as f:
    json.dump(erroneous, f)
