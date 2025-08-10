import pandas as pd
from pymatgen.core.structure import Structure


def clean_and_convert_temperature_string(temperature_string: str):
    """
    Clean experiment / transition temperature string parsed
    from mcif and convert to float, if possible.
    :param temperature_string: string from mcif keys
    _transition_temperature or _experiment_temperature
    :return: cleaned temperature float | None if string is not convertable
    """
    # Uncertainties in round brackets -> remove those and content in brackets
    # Also handle None case
    try:
        temperature_string = temperature_string.split("(")[0]
    except AttributeError:
        return None

    # Remove units in K
    temperature_string = temperature_string.replace("K", "")

    # Try conversion to float
    try:
        return float(temperature_string)
    except ValueError:
        return None


def convert_citation_year_string(citation_year_string: str):
    """
    Convert citation year string parsed
    from mcif to int, if possible.
    :param citation_year_string: string from mcif key _citation_year
    :return: cleaned year int | None if string is not convertable
    """
    try:
        return int(citation_year_string)
    except (ValueError, TypeError):
        return None


def get_crystallographic_primitive(structure: Structure):
    """
    Convert magnetic structure to its crystallographic primitive.
    :param structure: pymatgen Structure object
    :return: non-magnetic primitive pymatgen Structure object
    """
    structure.remove_site_property(property_name="magmom")
    return structure.get_primitive_structure(tolerance=0.05)


def choose_by_newest_publication_or_pick_lowest_index(df: pd.DataFrame):
    """
    Out of df with sanitized (integer) citation year column, choose newest entry or,
    if not possible, return entry with lowest index
    :param df: pd.DataFrame
    :return: df index value
    """
    if True in df["citation_year_san"].notna().values:
        citation_year_present = df.loc[df["citation_year_san"].notna()]
        return citation_year_present.loc[citation_year_present["citation_year_san"] ==
                                         citation_year_present["citation_year_san"].values.max()].index[0]
    else:
        return df.index[0]


def pick_only_one_or_choose_by_newest_publication_or_pick_lowest_index(df: pd.DataFrame):
    """
    Out of df with sanitized (integer) citation year column, return index of first entry if only one,
    otherwise apply choose_by_newest_publication_or_pick_lowest_index()
    :param df: pd.DataFrame
    :return: df index value
    """
    if len(df) == 1:
        return df.index[0]

    # If multiple entries have highest transition temperature, choose by newest publication, after that lowest index
    else:
        return choose_by_newest_publication_or_pick_lowest_index(df=df)
