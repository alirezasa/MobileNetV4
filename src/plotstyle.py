"""Shared matplotlib style: one colour per normalisation, recessive axes, thin marks."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {'bn': '#2a78d6', 'ln': '#eb6834', 'none': '#1baf7a', 'gn1': '#6250d6'}
LABELS = {'bn': 'BatchNorm', 'ln': 'LayerNorm', 'none': 'No norm', 'gn1': 'GroupNorm(1)'}
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'lines.linewidth': 2, 'legend.frameon': False,
})
