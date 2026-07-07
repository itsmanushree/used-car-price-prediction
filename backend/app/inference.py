"""
inference.py
-------------
Rebuilds, at prediction time, the EXACT 69-column feature vector that the
Random Forest model was trained on (see notebooks/01, 02, 03).

Pipeline recap (done during training, replicated here for a single car):
1. High-cardinality categorical columns (brand, model, variant, color, city,
   etc.) were frequency-encoded: value -> (count of that value / total rows).
2. Low-cardinality categorical columns (fuel type, transmission, owner type,
   turbo/super charger, engine cc bucket) were one-hot encoded with
   drop_first=True.
3. Everything else was left as a plain numeric column.

Because most of a car's technical specs (engine size, dimensions, brake
type, etc.) are basically fixed for a given brand/model/variant, we don't
ask the user for them. Instead we look them up from `variant_lookup.csv`,
which stores the median/most common value of every technical spec for each
variant seen in the training data. The user only supplies the things that
genuinely differ between two cars of the exact same variant: kilometers
driven, registration year, owner type, city and color.
"""

import csv
import json
import warnings
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

# The model was trained on a pandas DataFrame (named columns). At inference
# time we intentionally pass a plain numpy array (same column order, no
# pandas dependency needed in production) which triggers a harmless
# UserWarning from scikit-learn. We silence just that one warning.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


# --------------------------------------------------------------------------
# Loading artifacts (done once, at import time)
# --------------------------------------------------------------------------

def _load_json(name):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_as_dicts(name):
    with open(DATA_DIR / name, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


FREQ_MAPS = _load_json("freq_maps.json")                 # high-card col -> {value: freq}
LOW_CARD_CATEGORIES = _load_json("low_card_categories.json")  # low-card col -> [categories]
GLOBAL_NUM_MEDIANS = _load_json("global_num_medians.json")

VARIANT_ROWS = _load_csv_as_dicts("variant_lookup.csv")
CITY_ROWS = _load_csv_as_dicts("city_lookup.csv")

FEATURE_COLUMNS = joblib.load(MODEL_DIR / "feature_columns.pkl")
MODEL = joblib.load(MODEL_DIR / "random_forest_model.pkl")

# Fast lookup indexes
VARIANT_BY_NAME = {row["variant_name"]: row for row in VARIANT_ROWS}
CITY_BY_NAME = {row["city_name_new"]: row for row in CITY_ROWS}

NUMERIC_VARIANT_COLS = [
    "Displacement", "Max Power", "Max Torque", "No of Cylinder",
    "Values per Cylinder", "Length", "Width", "Height", "Wheel Base",
    "Turning Radius", "No Door Numbers", "Cargo Volumn",
    "seating_capacity_new", "max_engine_capacity_new",
    "min_engine_capacity_new", "mileage_new", "top_features_count",
    "comfort_features_count", "interior_features_count",
    "exterior_features_count", "safety_features_count",
]

CATEGORICAL_VARIANT_COLS = [
    "brand_name", "model_new", "oem_name", "car_segment", "body_type_new",
    "bt", "fuel_type_new", "transmission_type_new", "engine_cc",
    "engine_capacity_new", "Engine Type", "Value Configuration",
    "Turbo Charger", "Super Charger", "Gear Box", "Drive Type",
    "Steering Type", "Front Brake Type", "Rear Brake Type", "Tyre Type",
    "Fuel Suppy System",
]


# --------------------------------------------------------------------------
# Dropdown helpers (used by the /api/* endpoints)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_brands():
    brands = sorted({row["brand_name"] for row in VARIANT_ROWS})
    return brands


@lru_cache(maxsize=None)
def get_models_for_brand(brand: str):
    models = sorted({row["model_new"] for row in VARIANT_ROWS if row["brand_name"] == brand})
    return models


@lru_cache(maxsize=None)
def get_variants_for_model(model: str):
    rows = [row for row in VARIANT_ROWS if row["model_new"] == model]
    rows.sort(key=lambda r: -float(r["listing_count"]))
    return [row["variant_name"] for row in rows]


@lru_cache(maxsize=1)
def get_cities():
    rows = sorted(CITY_ROWS, key=lambda r: -float(r["n"]))
    return [row["city_name_new"] for row in rows]


@lru_cache(maxsize=1)
def get_colors():
    freq = FREQ_MAPS["Color"]
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:15]
    colors = [c for c, _ in top]
    if "Other" not in colors:
        colors.append("Other")
    return colors


OWNER_TYPE_DISPLAY = {
    "first": "1st Owner",
    "second": "2nd Owner",
    "third": "3rd Owner",
    "fourth": "4th Owner",
    "fifth": "5th Owner",
    "unregistered car": "Unregistered Car",
}


def get_owner_types():
    cats = LOW_CARD_CATEGORIES["owner_type_new"]
    return [{"value": c, "label": OWNER_TYPE_DISPLAY.get(c, c.title())} for c in cats]


def get_variant_details(variant_name: str):
    row = VARIANT_BY_NAME.get(variant_name)
    if row is None:
        return None
    return {
        "variant_name": row["variant_name"],
        "brand_name": row["brand_name"],
        "model_new": row["model_new"],
        "fuel_type": row["fuel_type_new"],
        "transmission": row["transmission_type_new"],
        "body_type": row["bt"],
        "engine_cc_bucket": row["engine_cc"],
        "displacement_cc": row["Displacement"],
        "max_power_bhp": row["Max Power"],
        "mileage_kmpl": row["mileage_new"],
        "seating_capacity": row["seating_capacity_new"],
    }


# --------------------------------------------------------------------------
# Core feature-vector construction
# --------------------------------------------------------------------------

def _freq_encode(col_name: str, raw_value):
    return float(FREQ_MAPS.get(col_name, {}).get(str(raw_value), 0.0))


def _safe_float(value, fallback):
    try:
        if value is None or value == "":
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def build_feature_dict(variant_name: str, km_driven: float, model_year: int,
                        owner_type: str, city_name: str, color: str):
    variant_row = VARIANT_BY_NAME.get(variant_name)
    if variant_row is None:
        raise ValueError(f"Unknown variant: {variant_name}")

    city_row = CITY_BY_NAME.get(city_name)
    state = city_row["state"] if city_row else ""
    loc = city_row["loc"] if city_row else ""

    raw = {}
    # location + listing-specific fields
    raw["km"] = km_driven
    raw["km_driven"] = km_driven
    raw["model_year_new"] = model_year
    raw["owner_type_new"] = owner_type
    raw["city_name_new"] = city_name
    raw["state"] = state
    raw["loc"] = loc
    raw["Color"] = color
    raw["exterior_color"] = color

    # technical specs, from the variant lookup table
    for col in CATEGORICAL_VARIANT_COLS:
        raw[col] = variant_row.get(col, "")
    for col in NUMERIC_VARIANT_COLS:
        raw[col] = _safe_float(variant_row.get(col), GLOBAL_NUM_MEDIANS.get(col, 0.0))

    feature_dict = {}

    # 1. frequency-encoded (high cardinality) columns
    for col in FREQ_MAPS.keys():
        feature_dict[col] = _freq_encode(col, raw.get(col, ""))

    # 2. one-hot (low cardinality) columns -> "<col>_<category>"
    for col, categories in LOW_CARD_CATEGORIES.items():
        for cat in categories:
            dummy_name = f"{col}_{cat}"
            if dummy_name in FEATURE_COLUMNS:
                feature_dict[dummy_name] = 1.0 if str(raw.get(col)) == str(cat) else 0.0

    # 3. plain numeric passthrough columns
    for col in NUMERIC_VARIANT_COLS + ["km", "km_driven", "model_year_new"]:
        feature_dict[col] = _safe_float(raw.get(col), GLOBAL_NUM_MEDIANS.get(col, 0.0))

    return feature_dict


def predict_price(variant_name: str, km_driven: float, model_year: int,
                   owner_type: str, city_name: str, color: str) -> float:
    feature_dict = build_feature_dict(
        variant_name, km_driven, model_year, owner_type, city_name, color
    )
    row = [feature_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
    X = np.array([row], dtype=float)
    prediction = MODEL.predict(X)[0]
    return float(prediction)
