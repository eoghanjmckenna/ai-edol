"""
Shared plot style constants for the evaluation pipeline.

Colours (Okabe-Ito, colourblind-safe) and figure sizes for paper and
presentation output modes. All visualisation code imports from this module —
colours for synthetic/train/test data must never be hardcoded in individual
plot functions.
"""

# --- Colours (Okabe-Ito, colourblind-safe) ---
COLOUR_SYNTH = "#0072B2"   # blue
COLOUR_TRAIN = "#E69F00"   # amber
COLOUR_TEST  = "#009E73"   # green
COLOUR_VAL   = "#CC79A7"   # pink

LABEL_SYNTH = "Synthetic"
LABEL_TRAIN = "Train"
LABEL_TEST  = "Test"
LABEL_VAL   = "Val"

SPLIT_STYLES = {
    'synth': {'color': COLOUR_SYNTH, 'label': LABEL_SYNTH},
    'train': {'color': COLOUR_TRAIN, 'label': LABEL_TRAIN},
    'test':  {'color': COLOUR_TEST,  'label': LABEL_TEST},
    'val':   {'color': COLOUR_VAL,   'label': LABEL_VAL},
}

# --- Paper mode ---
PAPER = {
    'figsize_single': (6, 4),
    'figsize_wide':   (10, 4),
    'figsize_2x2':    (10, 8),
    'figsize_2x3':    (12, 8),
    'dpi': 300,
    'rc': {
        'font.size': 9,
        'lines.linewidth': 1.5,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
    },
}

# --- Presentation mode ---
PRESENTATION = {
    'figsize_single': (10, 6),
    'figsize_wide':   (16, 6),
    'figsize_2x2':    (16, 10),
    'figsize_2x3':    (20, 12),
    'dpi': 150,
    'rc': {
        'font.size': 16,
        'lines.linewidth': 2.5,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
    },
}

STYLES = {'paper': PAPER, 'presentation': PRESENTATION}

# --- SDC threshold ---
# Suppress real-data statistics where n_households < this
SDC_MIN_HOUSEHOLDS = 10
