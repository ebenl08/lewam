"""Shared plot style for the LeWAM planner comparison: one hue per planner family, shades within a family.

Surface, ink and colormaps copied from playground/stable-worldmodel (scripts/latency/plot_latency.py,
src/utils/plot/plot_goal_offset.py, src/utils/plot/plot_sr_diff.py); no import from that project.
Families: LeWAM = blue, CEM + LeFlow → CEM = orange, iCEM + LeFlow → iCEM = green. Window variants of one planner
share its colour: w=1 hollow / dashed / hatched, w=H filled / solid.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
SEQUENTIAL = LinearSegmentedColormap.from_list('sr_blue', ['#eef4fc', '#2a78d6', '#0d3a73'])
DIVERGING = LinearSegmentedColormap.from_list('diff', ['#e34948', '#f0efec', '#2a78d6'])

FAMILY_COLORS = {
    'native': '#1f5bb5', 'grid_wH': '#2a78d6', 'grid_w1': '#6ea8ee',     # LeWAM: blue, dark = as published
    'cem_300x30': '#eb6834', 'cem': '#eb6834', 'leflow_cem': '#a8400f',  # CEM family: orange (300x30 has its own plot)
    'icem_faithful': '#1baf7a', 'leflow_icem': '#0b6b47',                # iCEM family: green
}


def base_arm(arm):
    """Planner without its window suffix (cem_w1 -> cem); LeWAM grid arms keep theirs."""
    return arm[:-3] if arm.endswith(('_w1', '_wH')) and not arm.startswith('grid') else arm


def color(arm):
    return FAMILY_COLORS[base_arm(arm)]


def is_w1(arm):
    return arm.endswith('_w1')


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    })
