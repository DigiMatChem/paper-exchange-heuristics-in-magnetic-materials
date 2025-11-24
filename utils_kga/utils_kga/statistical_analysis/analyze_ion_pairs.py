from collections import Counter
from math import log10
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go


# List tailored for larger bond angles between TM octahedra in MAGNDATA, can be customized (ion_list param below)
sort_by_electron_configuration = [
    ("V", 3), ("Os", 6),
    ("Cr", 3), ("Mn", 4), ("Ru", 5), ("Os", 5),
    ("Mn", 3),
    ("Ru", 4),
    ("Mn", 2), ("Fe", 3), ("Co", 4),
    ("Ir", 4),
    ("Fe", 2), ("Co", 3),
    ("Co", 2), ("Ni", 3),
    ("Ni", 2),
    ("Cu", 2)
]

# Simple positive colorscale for occurrences including None
viridis = pc.get_colorscale('Viridis')
cs_ion_pair_occus_lin = [(0.0, "#929292")] + [(v[0] * 0.999 + 0.001, v[1]) for v in viridis]
cs_ion_pair_occus_log = [(0.0, "#929292")] + [(e, v[1]) for e, v in zip([2**f for f in range(-9, 1)], viridis)]


def create_custom_diverging_color_scale(endpoint1: tuple[int, int, int] = (49,54,149), #(0, 159, 227), #(5, 82, 104), 
                                        endpoint2: tuple[int, int, int] = (165,0,38), #(210, 0, 30), #(139, 24, 17),
                                        midpoint: tuple[int, int, int] = (195, 195, 195), #(255,255,141), #(256, 256, 256)
                                        )-> list:
    """Vals in RGB."""
    def interpolate_rgb(color1, color2, t):
        return tuple([
            int(color1[i] + (color2[i] - color1[i]) * t)
            for i in range(3)
        ])
    steps = 6
    scale = [interpolate_rgb(endpoint1, midpoint, i / (steps - 1)) for i in range(steps)] + [interpolate_rgb(midpoint, endpoint2, i / (steps - 1)) for i in range(steps)][1:]
    return [[i/10, f"rgb{rgb}"] for rgb, i in zip(scale, range(11))]


# Log colorscale for FM/AFM ratios including None
# ocs = pc.get_colorscale("RdBu_r")
ocs = create_custom_diverging_color_scale()
ratio_min = -1.2
ratio_min2 = -1.0
ratio_max = 1.0
cs_ion_pair_occus_ratio = [[ratio_min, "#FFFFFF"], ] + [[i / 10, ocs[i_idx][1]] for i_idx, i in
                                                        enumerate(range(-10, 12, 2))]
cs_ion_pair_occus_ratio = [[round((c[0] - ratio_min) / (ratio_max - ratio_min), 6), c[1]] for c in
                           cs_ion_pair_occus_ratio]


def get_ion_pair_occus(df: pd.DataFrame, n_lattice_points: int, normalize_bool: bool = False) -> dict:
    ion_pairs = [(row["site_element"], 
                 int(row["site_oxidation"]), 
                  row["site_to_element"], 
                  int(row["site_to_oxidation"])) for _, row in df.iterrows()]
    ion_pairs_dict = Counter(ion_pairs)

    # Weigh off-diagonal elements same as diagonal
    ion_pairs_dict_c = ion_pairs_dict.copy()
    for ip, c in ion_pairs_dict_c.items():
        if not (ip[0] == ip[2] and ip[1] == ip[3]):
            ion_pairs_dict[ip] = 2 * c

    if normalize_bool:
        abs_ion_pairs = {k: v / sum(list(ion_pairs_dict.values())) for k, v in ion_pairs_dict.items()}
    else:
        abs_ion_pairs = {k: v / n_lattice_points for k, v in ion_pairs_dict.items()}

    return abs_ion_pairs


def plot_ion_pair_occurrences(occus: dict, log: bool = False, ion_list: list | None = None) -> dict:
    if not ion_list:
        ion_list = sort_by_electron_configuration
    figure_dict = {}
    for mag_type, occu_dict in occus.items():
        x, y, z = [], [], []
        for el1, ox1 in ion_list:
            for el2, ox2 in ion_list:
                x.append(f"{el1}{ox1}+")
                y.append(f"{el2}{ox2}+")
                if (el1, ox1, el2, ox2) in occu_dict:
                    z.append(occu_dict[(el1, ox1, el2, ox2)])
                else:
                    z.append(0)
        colorscale = cs_ion_pair_occus_log if log else cs_ion_pair_occus_lin
        figure_dict[mag_type] = get_ion_pair_heatmap(x=x, 
                                                     y=y, 
                                                     z=z,
                                                     zmax=max(v2 for v1 in occus.values() for v2 in v1.values()),
                                                     colorscale=colorscale,
                                                     tickval_len=len(ion_list))

    return figure_dict


def plot_ion_pair_occurrence_ratio(occus: dict,
                                   mag_interaction_types_to_compare: tuple[str, str] = ("fm", "afm"),
                                   ion_list: list | None = None) -> go.Figure:
    """Heatmap of ratio with logarithmic, diverging color scale."""
    m1, m2 = mag_interaction_types_to_compare
    assert m1 in occus
    assert m2 in occus

    if not ion_list:
        ion_list = sort_by_electron_configuration

    x, y, z= [[] for _ in range(3)]
    for el1, ox1 in ion_list:
        for el2, ox2 in ion_list:
            x.append(f"{el1}{ox1}+")
            y.append(f"{el2}{ox2}+")

            if (el1, ox1, el2, ox2) in occus[m1]:
                if (el1, ox1, el2, ox2) in occus[m2]:
                    orig = occus[m1][(el1, ox1, el2, ox2)] / occus[m2][(el1, ox1, el2, ox2)]
                    ratio = log10(orig)
                    if not -1 < ratio < 1:
                        # Log ratios outside of (0.1, 10) to log whether extreme points on scale only belong to "only AFM" a. "only FM"
                        print(f"ratio of {ratio} in {el1, ox1, el2, ox2}")
                        ratio = -1 if ratio < 0 else 1
                    z.append(ratio)
                else:
                    z.append(ratio_max)
            elif (el1, ox1, el2, ox2) in occus[m2]:
                z.append(ratio_min2)
            else:
                z.append(ratio_min)

    return get_ion_pair_heatmap(x=x,
                                y=y,
                                z=z,
                                zmax=max(z),
                                zmin=min(z),
                                colorscale=cs_ion_pair_occus_ratio,
                                tickval_len=len(ion_list),
                                is_log_ratio_plot=True)


def get_ion_pair_heatmap(x: list, 
                         y: list, 
                         z: list, 
                         colorscale: list, 
                         zmax: float, 
                         zmin: float = 0, 
                         tickval_len: int = len(sort_by_electron_configuration),
                         is_log_ratio_plot: bool = False
                         ) -> go.Figure:
    if is_log_ratio_plot:
        fig = go.Figure(
            go.Heatmap(x=x, y=y, z=z, colorscale=colorscale, zmax=zmax, zmin=zmin, 
                    colorbar=dict(
            tickmode="array",
            ticklen=0,
            tickvals=[-1.2, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0],
            ticktext=['None', '<=0.1', '0.18', '0.32', '0.56', '1.0', '1.8', '3.2', '5.6', '>=10'],
            ticks="outside",
            tickfont=dict(size=21)
        )))
    else:
        fig = go.Figure(
            go.Heatmap(x=x, y=y, z=z, colorscale=colorscale, zmax=zmax, zmin=zmin, 
                       colorbar=dict(tickfont=dict(size=21))))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(tickval_len)),
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(tickval_len)),
        ),
        width=700,
        height=700,
        autosize=True
    )
    fig.update_xaxes(dict(
        tickfont=dict(size=21, color="black"),),
        showline=True,
        mirror=True,
        linewidth=1.0,
        linecolor="black",)
    fig.update_yaxes(dict(
        tickfont=dict(size=21, color="black"),),
        showline=True,
        mirror=True,
        linewidth=1.0,
        linecolor="black",)
    return fig
