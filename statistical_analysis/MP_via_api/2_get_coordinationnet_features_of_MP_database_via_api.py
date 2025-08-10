import json
from monty.json import MontyEncoder
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from spglib import get_magnetic_symmetry_dataset

from utils_kga.coordination_features import CoordinationFeatures


mag_structures = pd.read_json("data/250512_unique_4559_mag_structs.json")

row_lists, erroneous = [], []

for mp_id, row in mag_structures.iterrows():
    structure = Structure.from_dict(row["structure"])
    # Attempt coordinationnet featurization
    try:
        cn_feat = CoordinationFeatures().from_structure(structure=structure,
                                                        env_strategy="simple",
                                                        guess_oxidation_states_from_composition=False,
                                                        include_edge_multiplicities=True)
        oxidation_states_origin = "BVA"
    except (ValueError, TypeError):
        try:
            cn_feat = CoordinationFeatures().from_structure(structure=structure,
                                                            env_strategy="simple",
                                                            guess_oxidation_states_from_composition=True,
                                                            include_edge_multiplicities=True)
            oxidation_states_origin = "pymatgen_composition_guess"
        except (ValueError, AssertionError) as e:
            erroneous.append((mp_id, str(e)))
            continue

    # Compute primitive magnetic structure and its space group type to arrive at n_lattice points of
    sga_mag = SpacegroupAnalyzer(structure=structure,
                                                symprec=1e-5,
                                                angle_tolerance=-1.0,
                                                )
    mag_symm_dataset = get_magnetic_symmetry_dataset(cell=sga_mag._cell,
                                                     symprec=1e-5,
                                                     angle_tolerance=-1.0,
                                                     mag_symprec=-1.0,
                                                     )
    try:
        prim_mag_lattice_volume = Lattice(mag_symm_dataset["primitive_lattice"]).volume
    except TypeError as e:
        print(mp_id, str(e))
        print(structure)
        print(mag_symm_dataset)
        erroneous.append((mp_id, str(e)))
        continue
    n_mag_lattice_points = structure.volume / prim_mag_lattice_volume

    # Assert that integer number of magnetic lattice points
    assert round(n_mag_lattice_points, 2) % 1 == 0.0, (
            str(mp_id) + ": " + str(structure.volume + " " + str(prim_mag_lattice_volume)))

    row_lists.append({
        "mp_id": mp_id,
        "formula": structure.formula,
        "structure_dict": structure.as_dict(),
        "coordination_features": cn_feat,
        "oxidation_states_origin": oxidation_states_origin,
        "n_mag_lattice_points": round(n_mag_lattice_points, 0),
    })

df = pd.DataFrame.from_records(row_lists)
df.set_index("mp_id", inplace=True, drop=True)


with open("data/unique_4559_cnfeat_MP_db_from_api.json", "w") as f:
    json.dump(df, f, cls=MontyEncoder)

# Save log of erroneous structures
with open("data/erroneous_structures_log_get_coordinationnet_features_of_MP_database.json", "w") as f:
    json.dump(erroneous, f)
