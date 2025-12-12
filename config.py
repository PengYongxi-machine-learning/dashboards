"""
Configuration file for Streamlit Task 7 Dashboard
All paths updated for your Windows working directory.
"""

import os

############################################################################################################
# How to run Streamlit
#
# In terminal / PowerShell:
#
# cd "C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\src"
# python -m streamlit run app.py
############################################################################################################

# =====================================================================
# 🗂 BASE DIRECTORY FOR ALL DATA
# =====================================================================
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 🌞 TASK 3 OUTPUTS (INVERTER RESULTS)
# Each plant has its own results folder containing *_results.pkl files.
# =====================================================================

PLANT_CONFIG = {
    "Plant 1": {
        # Folder containing 1BY6WEcLGh8j5v7_results.pkl, ALL_INVERTER_RESULTS.pkl, etc.
        "results_folder": os.path.join(DATA_ROOT, "01 Plant1_Inverter_Models"),
        "task3_results_glob": "*_results.pkl",

        # Cleaned CSVs (one per inverter) – used by fast retrain + Task 2
        "clean_root": os.path.join(DATA_ROOT, "00 Excel clean file", "Plant 1"),

        # Legacy naming (kept for clarity; same path)
        "daily_clean_folder": os.path.join(DATA_ROOT, "00 Excel clean file", "Plant 1"),
    },

    "Plant 2": {
        "results_folder": os.path.join(DATA_ROOT, "02 Plant2_Inverter_Models"),
        "task3_results_glob": "*_results.pkl",

        "clean_root": os.path.join(DATA_ROOT, "00 Excel clean file", "Plant 2"),
        "daily_clean_folder": os.path.join(DATA_ROOT, "00 Excel clean file", "Plant 2"),
    },
}

# =====================================================================
# 🌤 FEATURE CONFIGURATION (for interactive predictions – Task 7)
# These names must match the dataframe columns used in utils.add_engineered_features()
# =====================================================================

FEATURES_CONFIG = [
    {"name": "IRRADIATION_CLEAN", "label": "Irradiation (W/m²)", "min": 0,  "max": 1200, "default": 600,   "step": 10},
    {"name": "AMBIENT_TEMPERATURE", "label": "Ambient Temp (°C)", "min": -5, "max": 50,   "default": 20,    "step": 1},
    {"name": "MODULE_TEMPERATURE",  "label": "Module Temp (°C)",  "min": 0,  "max": 80,   "default": 35,    "step": 1},
    {"name": "DAILY_YIELD_CLEAN",   "label": "Daily Yield (kWh)", "min": 0,  "max": 2000, "default": 200,   "step": 5},
    {"name": "TOTAL_YIELD_CLEAN",   "label": "Total Yield (kWh)", "min": 0,  "max": 1e7,  "default": 50000, "step": 100},
    {"name": "NUM_OPT",             "label": "# Optimal Ops",     "min": 0,  "max": 144,  "default": 100,   "step": 1},
    {"name": "NUM_SUBOPT",          "label": "# Sub-optimal Ops", "min": 0,  "max": 144,  "default": 10,    "step": 1},
    {"name": "hour_cos",            "label": "cos(hour)",         "min": -1, "max": 1,    "default": 0.5,   "step": 0.01},
    {"name": "hour_sin",            "label": "sin(hour)",         "min": -1, "max": 1,    "default": 0.0,   "step": 0.01},
]

FEATURE_NAMES = [f["name"] for f in FEATURES_CONFIG]

# =====================================================================
# 🤖 MODEL OPTIONS (fast models used in Predictions tab)
# Names must match utils._train_models_for_target()
# =====================================================================

MODEL_NAMES = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "RandomForest",
    "MLPRegressor",
    "NeuralNet",      # <--------------------------------------------------------------------------------------- add this
]
# =====================================================================
# 📊 TASK 4 — CLASSIFICATION OUTPUTS (ALE + SVM)

# =====================================================================

TASK4_FOLDER = os.path.join(DATA_ROOT, "03 ALE SVM Decision")
TASK4_RESULTS_GLOB = "results_Run_*.pkl"

# =====================================================================
# 🌳 TASK 5 — RANDOM FOREST FEATURE IMPORTANCE
# Stored in:

# =====================================================================

TASK5_FOLDER = os.path.join(DATA_ROOT, "04 AC DC")
TASK5_RF_AC_DC = os.path.join(TASK5_FOLDER, "feature_importance_results_AC_DC.pkl")
TASK5_RF_SIMPLE = os.path.join(TASK5_FOLDER, "feature_importance_resultsRF.pkl")

# =====================================================================
# 📈 TASK 6 — LSTM FORECASTING MODELS
# Stored in:

# =====================================================================

TASK6_FOLDER = os.path.join(DATA_ROOT, "06 LSTM Forecasting Model")
TASK6_LSTM_PKL = os.path.join(TASK6_FOLDER, "LSTM Forecasting Model.pkl")

# =====================================================================
# End of config
# =====================================================================
