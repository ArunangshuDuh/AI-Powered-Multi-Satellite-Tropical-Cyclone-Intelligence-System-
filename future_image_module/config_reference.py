"""
Tropical Cyclone AI Backend - Configuration
All shared constants, paths, and mappings.
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
PRECOMPUTED_DIR = PROJECT_ROOT / "precomputed"

# Ensure directories exist
for d in [RAW_DIR, CACHE_DIR, CHECKPOINTS_DIR, PRECOMPUTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Image dimensions
# ──────────────────────────────────────────────
IMG_SIZE = 64          # Native training resolution (64×64)
DISPLAY_SIZE = 256     # Upscaled for API PNG responses
NUM_CHANNELS = 5       # IR1, WV, PMW_proxy, VIS, VIS_mask

# ──────────────────────────────────────────────
# Category mapping (matches API contract EXACTLY)
# category_code: 0=TD, 1=TS, 2=Cat1, 3=Cat2, 4=Cat3, 5=Cat4, 6=Cat5
# ──────────────────────────────────────────────
CATEGORY_MAP = {
    0: "TD",
    1: "TS",
    2: "Cat1",
    3: "Cat2",
    4: "Cat3",
    5: "Cat4",
    6: "Cat5",
}
CATEGORY_REVERSE = {v: k for k, v in CATEGORY_MAP.items()}
NUM_CATEGORIES = 7

# Wind speed thresholds for category assignment (in knots)
# Category is determined by max sustained wind speed:
#   TD: < 34, TS: 34-63, Cat1: 64-82, Cat2: 83-95,
#   Cat3: 96-112, Cat4: 113-136, Cat5: >= 137
WIND_THRESHOLDS = [0, 34, 64, 83, 96, 113, 137]


def wind_to_category_code(wind_kt: float) -> int:
    """Convert wind speed in knots to category code (0-6)."""
    for i in range(len(WIND_THRESHOLDS) - 1, -1, -1):
        if wind_kt >= WIND_THRESHOLDS[i]:
            return i
    return 0


def wind_to_category(wind_kt: float) -> str:
    """Convert wind speed in knots to category string."""
    return CATEGORY_MAP[wind_to_category_code(wind_kt)]


# ──────────────────────────────────────────────
# Demo storms (ATCF ID format)
# ──────────────────────────────────────────────
DEMO_STORMS = {
    "WP312013": {"name": "HAIYAN", "basin": "WPAC", "year": 2013},
    "AL122005": {"name": "KATRINA", "basin": "ATL", "year": 2005},
    "AL112017": {"name": "IRMA", "basin": "ATL", "year": 2017},
    "EP202015": {"name": "PATRICIA", "basin": "EPAC", "year": 2015},
    "AL052019": {"name": "DORIAN", "basin": "ATL", "year": 2019},
}

# IBTrACS SID prefixes for matching (basin+number+year format in IBTrACS)
STORM_IBTRACS_SIDS = {
    "WP312013": "2013314N07127",  # Haiyan
    "AL122005": "2005236N23285",  # Katrina
    "AL112017": "2005236N23285",  # Irma - will be resolved from IBTrACS
    "EP202015": "2015293N15291",  # Patricia
    "AL052019": "2019236N22309",  # Dorian
}

# ──────────────────────────────────────────────
# Training hyperparams
# ──────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
BACKBONE_LR_FACTOR = 0.1
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
LAMBDA_REGRESSION = 0.1  # Starting value, adjusted after epoch 1

# ──────────────────────────────────────────────
# GRU params
# ──────────────────────────────────────────────
GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS = 2
GRU_SEQ_LENGTH = 4       # 4 frames ≈ 12 hours at ~3h intervals
GRU_EPOCHS = 50
GRU_LR = 1e-3

# ──────────────────────────────────────────────
# Model dimensions
# ──────────────────────────────────────────────
NUM_META_FEATURES = 4     # month, lat, lon, hour
BACKBONE_FEAT_DIM = 1280  # EfficientNet-B0 output features
REGRESSION_OUTPUTS = 2    # wind_kt, pressure_mb

# ──────────────────────────────────────────────
# Rapid Intensification threshold
# ──────────────────────────────────────────────
RI_THRESHOLD_KT = 30  # >= 30 knots increase in 24 hours

# ──────────────────────────────────────────────
# Data URLs
# ──────────────────────────────────────────────
IBTRACS_CSV_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
HURSAT_BASE_URL = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/v06/"

# ──────────────────────────────────────────────
# Normalization stats (populated during preprocessing)
# ──────────────────────────────────────────────
# Will be saved/loaded as part of the cached tensor file
NORM_STATS_FILE = CACHE_DIR / "norm_stats.pt"

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
