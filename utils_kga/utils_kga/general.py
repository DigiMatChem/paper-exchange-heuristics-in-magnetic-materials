""" General util functions used within the whole project. """
from math import pi
import numpy as np
from plotly.graph_objs import Figure
from numpy.typing import ArrayLike
from pymatgen.core import Structure

from utils_kga.coordination_features import CoordinationFeatures


def get_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_angle_between(v1, v2):
    """ Returns the angle between vectors 'v1' and 'v2' in degrees."""
    v1_u = get_unit_vector(v1)
    v2_u = get_unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / pi


def pretty_plot(figure: Figure, width: int = 2500):
    """
    Changes layout of plotly figure similarly to pymatgen's
    pretty_plot function for matplotlib figures.
    :param figure: a plotly.graph_objects Figure object
    :param width: width in pixels
    :return: a plotly.graph_objects Figure object
    """

    tickfont = int(width / 100 * 0.9)
    titlefont = int(width / 100 * 1.0)

    """
    golden_ratio = (math.sqrt(5) - 1) / 2
    if not height:
        height = int(width * golden_ratio)
    """
    figure.update_layout(
        plot_bgcolor='rgba(255,255,255,1) ',
        paper_bgcolor='rgba(255,255,255,1) ',
        titlefont=dict(size=titlefont, color="black"),
        font_family="Arial",
        legend=dict(font=dict(size=tickfont, color="black"))
    )

    figure.update_xaxes(dict(
        titlefont=dict(size=titlefont, color="black"),
        tickfont=dict(size=tickfont, color="black"),
        ticks="outside",
        tickwidth=1.2,
        ticklen=10,
        showgrid=False,
        showline=True,
        mirror=True,
        linewidth=1.2,
        linecolor="black"
    ))
    figure.update_yaxes(dict(
        titlefont=dict(size=titlefont, color="black"),
        tickfont=dict(size=tickfont, color="black"),
        ticks="outside",
        tickwidth=1.2,
        ticklen=10,
        showgrid=False,
        showline=True,
        mirror=True,
        linewidth=1.2,
        linecolor="black",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=2,
    ))

    return figure


def get_coordination_features_and_supercell_coordination_features(
        structure: Structure, supercell_matrix: ArrayLike = 2):
    """ Utility function to create CoordinationFeatures object of given structure
    and its superstructure as per supercell_matrix parameter
    (usage see make_supercell Structure method in pymatgen).
    Used in tests."""
    super_structure = structure.make_supercell(scaling_matrix=supercell_matrix, in_place=False)

    cn_feat = CoordinationFeatures().from_structure(structure, include_edge_multiplicities=True)
    super_cn_feat = CoordinationFeatures().from_structure(super_structure,
                                                          guess_oxidation_states_from_composition=True,
                                                          include_edge_multiplicities=True)
    return super_structure, cn_feat, super_cn_feat

