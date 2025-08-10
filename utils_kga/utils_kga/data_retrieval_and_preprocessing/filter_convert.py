"""
Utilities for magnetic structure converting and filtering.
"""
import numpy as np
from pymatgen.core import Structure


OCCU_THR = 0.9  # Global threshold for ALL partial occupancies in a structure to be converted to 1.0
DIST_THR = 0.6  # Global threshold to scan for unphysical distances (hidden partial occupancies) in Angstrom

# Global to define possible lattice centering types and the number of lattice points in them
CENTERING_TYPES_AND_DIVISORS = {"P": 1,
                                "A": 2,
                                "B": 2,
                                "C": 2,
                                "I": 2,
                                "R": 3,
                                "F": 4}

TRANSFORMATION_CHARS = {"a", "b", "c", " ", "-", "+", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/"}


def convert_deuterium(structure):
    """ Convert D to H for later handling in packages like pymatgen."""
    struct_dict = structure.as_dict()
    for site in struct_dict["sites"]:
        if site["label"] in ["D", "D0+"]:
                site["label"] = "H"
        for specie in site["species"]:
            if specie["element"] in ["D", "D0+"]:
                specie["element"] = "H"
                specie["oxidation_state"] = None
    return Structure.from_dict(struct_dict)


def convert_filter_part_occus(structure):
    """Check for partial occupancies,
    convert to 1.0 if all partial occupations are greater than OCCU_THR,
    otherwise return None."""
    for site in structure.sites:
        if not site.is_ordered:
            species_list = site.as_dict()["species"]
            for s in species_list:
                if s["occu"] < OCCU_THR:
                    return None
    struct_dict = Structure.as_dict(structure)
    for site in struct_dict["sites"]:
        for specie in site["species"]:
            specie["occu"] = 1.0
    return Structure.from_dict(struct_dict)


def has_hidden_splits(structure):
    """Check for unphysically low distances (usually hidden split positions)."""
    dist_matrix = structure.distance_matrix
    unique_dists = set(dist_matrix.flatten())
    unique_dists.remove(0.00)
    if min(unique_dists) < DIST_THR:
        return True
    return False


def extract_msg_and_centering(mcif_dict: dict,
                              key: str) -> dict | None:
    """
    Extract magnetic space group type and its centering from the mcif.
    :param mcif_dict: dict extracted from pymatgens's CifParser
    :param key: mcif key (variable because msg name mixed with its number in some entries).
    :return: dictionary of msg and centering or None (if unsuccessful)
    """
    try:
        msg = mcif_dict[key]
    except KeyError:
        return None

    try:
        # Data cleaning as msg sometimes written in ''
        # -> removal ok if first character (otherwise it may encode symmetry)
        # Also removal of leading space
        msg = msg.removeprefix("'")
        msg = msg.removeprefix(" ")
        centering = msg[0]
    except IndexError:
        return None

    try:
        # Test if valid centering
        assert centering in CENTERING_TYPES_AND_DIVISORS
        return {"msg_type": msg,
                "centering": centering,
                "n_lattice_points": CENTERING_TYPES_AND_DIVISORS[centering]}
    except AssertionError:
        return None


def get_number_of_lattice_points(md_id: str, mcif_dict: dict, centering_divisor: int) -> int | Exception:
    """
    Calculate number of lattice points in the magnetic cell as given in the mcif:
    1. Extract transformation matrix to standard setting from its string representation in the mcif
    2. Calculate its determinant -> volume change
    3. Divide centering divisor from centering type by determinant to get true number of lattice points
    in the current setting.
    :param md_id: MAGNDATA id (solely necessary to manually correct wrong transformation matrix of entry 1.707
    (as per May 2024)).
    :param mcif_dict: dict extracted from pymatgens's CifParser
    :param centering_divisor: n(lattice points) solely from centering type as extract
    by extract_msg_and_centering()
    :return: number of lattice points in setting as in mcif
    """
    if md_id != "1.707":
        trans_string = mcif_dict["_space_group_magn.transform_BNS_Pp_abc"]

        operation_strings = trans_string.split(";")[0]  # Remove origin shift substring
        # Remove whitespaces later because of possible fraction handling (do not know if contains whole numbers)
        operation_strings = operation_strings.split(",")
        # Sanity check: new basis of len(3)
        assert len(operation_strings) == 3
        # Sanity check: only allowed characters in "matrix"
        for ops in operation_strings:
            assert set([char for char in ops]).issubset(TRANSFORMATION_CHARS)

        # Extract matrix components from operation strings and add to transformation matrix
        atomic_operation_strings = []
        for ops in operation_strings:
            atomic_op = []
            minus_strings = ops.split("-")
            # Re-add the minus string - pay attention to first position
            #  If first entry starts with minus, "" will be first str, else entry will be first str, do not add minus
            minus_strings = [minus_strings[0]] + ["-" + st for st_idx, st in enumerate(minus_strings) if st_idx != 0]
            if minus_strings[0] == "":
                minus_strings = minus_strings[1:]

            for ms in minus_strings:
                plus_strings = ms.split("+")
                # Remove unnecessary plus signs
                if plus_strings[0] == "":
                    plus_strings = plus_strings[1:]
                atomic_op.extend(plus_strings)
            atomic_operation_strings.append(atomic_op)

        # Init transformation matrix
        transformation_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Map unit cell vector chars to respective position in transformation matrix
        basis_map = {"a": 0, "b": 1, "c": 2}

        for col_idx, a_ops_list in enumerate(atomic_operation_strings):
            for atomic_operation in a_ops_list:
                for basis_vector, basis_idx in basis_map.items():
                    if basis_vector in atomic_operation:
                        # There are different "matrix" string notations, see e.g. entries 0.110 vs. 0.3
                        subatomic_bits = atomic_operation.split(basis_vector)
                        if subatomic_bits[0] == "":  # Corresponding to atomic operation string starting with basis vector
                            if subatomic_bits[1] == "":  # Directly corresponding to basis vector
                                assert len(subatomic_bits) == 2
                                coeff_string = "1"
                            else:  # Corresponding to cases like "a/3"
                                coeff_string = atomic_operation.replace(basis_vector, "1")
                        elif subatomic_bits[1] == "":
                            if subatomic_bits[0] == "-":  # Directly corresponding to -basis vector
                                coeff_string = "-1"
                            else:
                                coeff_string = subatomic_bits[0]  # Corresponding to cases like "1/2 a"
                        elif subatomic_bits[0] == "-":  # Corresponding to cases like "-a/2"
                            coeff_string = atomic_operation.replace(basis_vector, "1")
                        else:  # Corresponding to cases like "2a/3"
                            coeff_string = "".join(subatomic_bits)

                        coeff = convert_to_float(coeff_string)
                        transformation_matrix[basis_idx][col_idx] = coeff
    else:  # Manual correction for entry 1.707 (as per May 2024 wrong transformation matrix)
        transformation_matrix = [[0.25, 0.25, -0.5], [0.25, -0.25, 0], [-0.5, -0.5, 0]]

    # Determine volume change by transformation
    delta_vol = np.linalg.det(transformation_matrix)

    # Determine true centering divisor
    n_lattice_points = round(centering_divisor / delta_vol, 6)  # floating point imprecision
    # Sanity check input -> only integer number of lattice points
    assert n_lattice_points % 1 == 0, (f"Non-integer volume change (true centering divisor = "
                                       f"{centering_divisor} / {delta_vol} = {n_lattice_points})")

    return n_lattice_points


def convert_to_float(frac_str):
    """ Utility function to extract float from fractional number string representation. """
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)

        return whole - frac if whole < 0 else whole + frac
