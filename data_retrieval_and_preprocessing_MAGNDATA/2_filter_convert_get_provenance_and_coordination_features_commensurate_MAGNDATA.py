"""
Starting from 1968 (minus 3 empty mcifs) commensurate MAGNDATA entries (9th Jan 24), this is a script to
- exclude structures
    - with failing pymatgen CifParser
    - that contain at least one partial occupancy with occu < 0.9 (non-convertable)
    - that contain unphysically low distances
    - that are not ordered (contained in first point, but extra mention)
- convert structures
    - with Deuterium to Hydrogen
    - with ALL partial occupancies >= 0.9 to ALL occus = 1.0
    ! for compatibility with pymatgen classes
- create CoordinationFeatures object
- get mcif date and temperature information
- summarize the above and additional info (contains_REs, composition, collinear magnetic ordering, ...) in a dataframe
"""
import json
from monty.json import MontyEncoder
import os
import pandas as pd
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser

from utils_kga.data_retrieval_and_preprocessing.filter_convert import *
from utils_kga.coordination_features import CoordinationFeatures


path_to_mcifs = "./mcifs"
mcif_dir = os.listdir(path_to_mcifs)

# Log dict for summarizing invalid mcif files, invalid structures, unphysical structures etc.
erroneous = {
    "mcif_content_assertion_error": [],
    "structure_not_implemented_error": [],
    "structure_value_error": [],
    "structure_attribute_error": [],
    "structure_has_low_partial_occupancies": [],
    "structure_has_hidden_splits": [],
    "structure_has_missing_elements": [],
    "structure_coordinationnet_fails": [],
    "mcif_non_integer_number_of_lattice_points": []
}

# Create df for all structures, provenance and metadata
df = pd.DataFrame(columns=[
    "md_id", "mag_structure", "composition", "contains_REs", "collinear_ordering", "total_moment_per_supercell",
    "citation_year", "audit_creation_date", "transition_temperature", "experiment_temperature", "msg_type",
    "centering", "n_lattice_points", "coordination_features", "oxidation_states_origin", "occupations_converted"])

for mcif in mcif_dir:
    occus_converted = False

    # Parse mcif content
    try:
        mcif_content = CifParser(filename=os.path.join(path_to_mcifs, mcif))
    except AssertionError as e:
        erroneous["mcif_content_assertion_error"].append((mcif, str(e)))
        continue

    # Parse structures fr. mcif
    try:
        structures = mcif_content.parse_structures(primitive=False, check_occu=True)
        if len(structures) != 1:
            print(mcif, len(structures))
    except NotImplementedError as e:
        erroneous["structure_not_implemented_error"].append((mcif, str(e)))
        continue
    except ValueError as e:
        erroneous["structure_value_error"].append((mcif, str(e)))
        continue
    except AttributeError as e:
        erroneous["structure_attribute_error"].append((mcif, str(e)))
        continue

    struct = structures[0]
    composition = Composition(struct.formula)

    # Convert Deuterium to Hydrogen if required
    if "D" in [el.symbol for el in composition.elements]:
        struct = convert_deuterium(struct)

    # Convert / filter out partial occupations of non-magnetic ions if required
    if not struct.is_ordered:
        struct = convert_filter_part_occus(struct)
        if not struct:
            erroneous["structure_has_low_partial_occupancies"].append((mcif, "has_low_partial_occupancies"))
            continue
        occus_converted = True

    # Check for unphysically low distances (hidden split positions)
    if has_hidden_splits(structure=struct):
        erroneous["structure_has_hidden_splits"].append((mcif, "has_hidden_splits"))
        continue

    # Attempt coordinationnet featurization
    try:
        cn_feat = CoordinationFeatures().from_structure(structure=struct,
                                                        env_strategy="simple",
                                                        guess_oxidation_states_from_composition=False,
                                                        include_edge_multiplicities=True)
        oxidation_states_origin = "BVA"
    except (ValueError, TypeError):
        try:
            cn_feat = CoordinationFeatures().from_structure(structure=struct,
                                                            env_strategy="simple",
                                                            guess_oxidation_states_from_composition=True,
                                                            include_edge_multiplicities=True)
            oxidation_states_origin = "pymatgen_composition_guess"
        except (ValueError, AssertionError) as e:
            erroneous["structure_coordinationnet_fails"].append((mcif, str(e)))
            continue

    # Renew (maybe changed) composition, assert if rare earth in structure
    composition = Composition(struct.formula)
    re_bool = composition.contains_element_type("rare_earth_metal")

    # Get collinear mag ordering (use results on non-coll. structures with care )
    mag_ordering = CollinearMagneticStructureAnalyzer(struct).ordering.value
    total_moment = round(sum(list(CollinearMagneticStructureAnalyzer(struct).magmoms)), 6)

    df_dict = {
        "md_id": mcif.removesuffix(".mcif"),
        "mag_structure": struct,
        "composition": composition.formula,
        "contains_REs": re_bool,
        "collinear_ordering": mag_ordering,
        "total_moment_per_supercell": total_moment,
        "coordination_features": cn_feat,
        "oxidation_states_origin": oxidation_states_origin,
        "occupations_converted": occus_converted
        }

    # Get provenance and experiment data from mcif
    mcif_dict = list(mcif_content.as_dict().values())[0]

    for key_string in ["_citation_year", "_audit_creation_date", "_transition_temperature", "_experiment_temperature"]:
        try:
            df_dict[key_string.removeprefix("_")] = mcif_dict[key_string]
        except KeyError:
            df_dict[key_string.removeprefix("_")] = None

    # Extract info about magnetic space group type and centering
    # Calculate volume change by transformation to standard setting to determine number of lattice points in cell
    msg_info = extract_msg_and_centering(mcif_dict=mcif_dict, key="_space_group_magn.name_BNS")
    if not msg_info:
        msg_info = extract_msg_and_centering(mcif_dict=mcif_dict, key="_space_group_magn.number_BNS")
        if not msg_info:
            msg_info = {"msg_type": None,
                        "centering": None,
                        "n_lattice_points": None}

    if isinstance(msg_info["n_lattice_points"], int):
        # Update number of lattice points as per volume change by transformation to standard setting
        n_lattice_points = get_number_of_lattice_points(md_id=mcif.removesuffix(".mcif"),
                                                        mcif_dict=mcif_dict,
                                                        centering_divisor=msg_info["n_lattice_points"])
        if isinstance(n_lattice_points, float) and n_lattice_points % 1 == 0:
            msg_info["n_lattice_points"] = n_lattice_points
        else:
            erroneous["mcif_non_integer_number_of_lattice_points"].append((mcif, str(n_lattice_points)))
            continue

    df_dict.update(msg_info)

    df_tmp = pd.DataFrame([df_dict])
    df = pd.concat([df, df_tmp], ignore_index=True)


# Make md_id index
df.set_index("md_id", inplace=True, drop=True)

# Sanity check: check occupations in df (if all structures with disordered sites filtered out / converted)
for idx, row in df.iterrows():
    structure_mag = row["mag_structure"]
    for site in structure_mag.sites:
        if not site.is_ordered:
            print("Disordered site in ", row["md_id"], site.species_string)

# Dump via JSON and MontyEncoder so Magmom objects can be recovered for later analysis
# Attention: nested recovery (json.load (df) / json.loads (structure)) required!
with open("data/df_commensurate_MAGNDATA.json", "w") as f:
    json.dump(df, f, cls=MontyEncoder)

# Save log of erroneous structures
with open("data/erroneous_structures_log_filter_convert_get_provenance_commensurate_MAGNDATA.json", "w") as f:
    json.dump(erroneous, f)
