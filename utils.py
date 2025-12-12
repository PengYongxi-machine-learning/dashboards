"""
Utility functions for the Streamlit dashboard.

Contains:
- Inverter discovery + CSV loading
- Engineered features for interactive predictions
- Lightweight "fast retrain" models for Task 7
- Readers for Task 3 *_results.pkl files
- Bias–variance proxies and NN loss curve extraction
- Wrappers around your full Task 3 training pipeline
"""

import os
import glob
import pickle
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from config import PLANT_CONFIG, MODEL_NAMES, FEATURE_NAMES


# ======================================================================
# SAFE HELPERS
# ======================================================================

def _plant_prefix(plant: str) -> str:
    """Prefix used in cleaned CSV filenames, e.g. Plant1_XXXX_clean.csv."""
    plant = plant.lower().strip()
    if plant == "plant 1":
        return "Plant1_"
    elif plant == "plant 2":
        return "Plant2_"
    return "Plant_"


def _safe_get_clean_root(cfg: dict) -> str | None:
    root = cfg.get("clean_root")
    if root is None or not os.path.isdir(root):
        return None
    return root


# ======================================================================
# ENGINEERED FEATURES
# ======================================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features needed for interactive predictions.

    - hour_cos, hour_sin (time of day)
    - DC/IRRA, AC/IRRA
    - Temp_Delta
    """
    df = df.copy()

    if "DATE_TIME" in df.columns:
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
        hour = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60.0
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)

    if "DC_CLEAN" in df.columns and "IRRADIATION_CLEAN" in df.columns:
        df["DC/IRRA"] = df["DC_CLEAN"] / (df["IRRADIATION_CLEAN"] + 1e-3)

    if "AC_CLEAN" in df.columns and "IRRADIATION_CLEAN" in df.columns:
        df["AC/IRRA"] = df["AC_CLEAN"] / (df["IRRADIATION_CLEAN"] + 1e-3)

    if "MODULE_TEMPERATURE" in df.columns and "AMBIENT_TEMPERATURE" in df.columns:
        df["Temp_Delta"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]

    return df


# ======================================================================
# LOAD CLEANED DATA
# ======================================================================

def get_inverters_for_plant(plant: str) -> List[str]:
    """
    Reads all cleaned CSVs and extracts inverter IDs from filenames like:
        Plant1_1BY6WEcLGh8j5v7_clean.csv
    """
    cfg = PLANT_CONFIG.get(plant)
    if cfg is None:
        raise ValueError(f"Unknown plant '{plant}'")

    clean_root = _safe_get_clean_root(cfg)
    if clean_root is None:
        raise FileNotFoundError(f"clean_root not configured for {plant}")

    prefix = _plant_prefix(plant)  # Plant1_ or Plant2_

    inverters: List[str] = []
    for fname in os.listdir(clean_root):
        if fname.startswith(prefix) and fname.endswith("_clean.csv"):
            inv_id = fname[len(prefix):-10]  # strip prefix + "_clean.csv"
            if inv_id:
                inverters.append(inv_id)

    if not inverters:
        raise RuntimeError(
            f"No inverter CSV files found for {plant} in folder:\n{clean_root}"
        )

    return sorted(inverters)


def _load_clean_inverter_dataframe(plant: str, inverter_id: str) -> pd.DataFrame:
    """
    Loads a cleaned CSV into a dataframe.
    """
    cfg = PLANT_CONFIG.get(plant)
    if cfg is None:
        raise ValueError(f"Unknown plant '{plant}'")

    clean_root = _safe_get_clean_root(cfg)
    if clean_root is None:
        raise FileNotFoundError(f"clean_root not configured for {plant}")

    prefix = _plant_prefix(plant)
    fname = f"{prefix}{inverter_id}_clean.csv"
    fpath = os.path.join(clean_root, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Cleaned CSV not found: {fpath}")

    df = pd.read_csv(fpath)
    if "DATE_TIME" in df.columns:
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df


# ======================================================================
# FEATURE BUILDING FOR FAST RETRAIN
# ======================================================================

def _build_X_y_for_interactive(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build X, y_dc, y_ac for lightweight "fast retrain" mode.

    Returns
    -------
    X : np.ndarray
    y_dc : np.ndarray
    y_ac : np.ndarray
    df_feat : pd.DataFrame (with all engineered features)
    """
    # Add engineered features (hour_cos, hour_sin, etc.)
    df = add_engineered_features(df)

    required_data_cols = ["DC_CLEAN", "AC_CLEAN"]
    for col in required_data_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in cleaned CSV")

    missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_feats:
        raise ValueError(
            f"Missing engineered feature columns {missing_feats} in dataframe. "
            f"Check add_engineered_features() and FEATURES_CONFIG."
        )

    use_cols = FEATURE_NAMES + required_data_cols
    df = df.dropna(subset=use_cols)

    X = df[FEATURE_NAMES].astype(float).values
    y_dc = df["DC_CLEAN"].astype(float).values
    y_ac = df["AC_CLEAN"].astype(float).values

    return X, y_dc, y_ac, df


def make_feature_input_defaults(stats: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Convert per-feature arrays into slider UI ranges.
    """
    cfg: Dict[str, dict] = {}

    for feat, values in stats.items():
        if values is None or len(values) == 0:
            continue
        mn = float(np.nanmin(values))
        mx = float(np.nanmax(values))
        default = float(np.nanmean(values))

        step = (mx - mn) / 200.0 if mx > mn else 0.01

        cfg[feat] = {
            "min": mn,
            "max": mx,
            "default": default,
            "step": step,
        }

    return cfg


def compute_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Universal feature-importance extractor.

    Supports:
    - Linear Regression / Ridge / Lasso → absolute coefficients
    - RandomForest → feature_importances_
    - MLPRegressor → gradient-based proxy (sum of absolute input weights)
    - Any other model → return {}
    """
    importance: Dict[str, float] = {}

    # ---------------------------
    # LINEAR MODELS (coef_)
    # ---------------------------
    if hasattr(model, "coef_"):
        coef = model.coef_
        coef = np.asarray(coef)
        if coef.ndim > 1:
            coef = coef[0]
        for f, c in zip(feature_names, coef):
            importance[f] = float(abs(c))
        return importance

    # ---------------------------
    # RANDOM FOREST
    # ---------------------------
    if hasattr(model, "feature_importances_"):
        for f, v in zip(feature_names, model.feature_importances_):
            importance[f] = float(v)
        return importance

    # ---------------------------
    # MLPRegressor (Neural Net)
    # approximate importance = sum |weights from input layer|
    # ---------------------------
    if hasattr(model, "coefs_") and len(model.coefs_) > 0:
        w = model.coefs_[0]  # shape (n_features, hidden_units)
        w_abs = np.sum(np.abs(w), axis=1)
        for f, v in zip(feature_names, w_abs):
            importance[f] = float(v)
        return importance

    # Unsupported
    return {}


# ======================================================================
# FAST RETRAIN MODE (Predictions tab)
# ======================================================================

def _train_models_for_target(X: np.ndarray, y: np.ndarray):
    """
    Train simple models for a single target (DC or AC).

    Model names are aligned with config.MODEL_NAMES.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(64, 64),
            learning_rate_init=0.001,
            max_iter=800,
            random_state=42,
        ),
    }

    fitted = {}
    metrics = {}

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        fitted[name] = mdl
        metrics[name] = {"rmse": rmse, "mae": mae}

    return fitted, metrics


def get_trained_models_for_inverter(plant: str, inverter_id: str) -> dict:
    """
    Lightweight retraining for the Predictions tab.

    Returns
    -------
    {
      "models_dc": {name: model},
      "models_ac": {name: model},
      "metrics_dc": {name: {rmse, mae}},
      "metrics_ac": {name: {rmse, mae}},
      "feature_stats": {feat: np.array([...])},
    }
    """
    df = _load_clean_inverter_dataframe(plant, inverter_id)
    X, y_dc, y_ac, df_feat = _build_X_y_for_interactive(df)

    if len(X) < 50:
        raise ValueError(
            f"Not enough samples ({len(X)}) in inverter '{inverter_id}' for retrain."
        )

    models_dc, metrics_dc = _train_models_for_target(X, y_dc)
    models_ac, metrics_ac = _train_models_for_target(X, y_ac)

    # ------------------------------------------------------------
    # NEW: expose a "NeuralNet" model in the fast tab by aliasing
    # the MLPRegressor model (so NeuralNet ~= MLPRegressor here).
    # ------------------------------------------------------------
    def add_neuralnet_alias(models: dict, metrics: dict):
        # try to find the MLP entry by name
        mlp_key = None
        for k in models.keys():
            if k.lower().startswith("mlp"):
                mlp_key = k
                break
        if mlp_key is None:
            return  # nothing to alias

        # only add if not already present
        if "NeuralNet" not in models:
            models["NeuralNet"] = models[mlp_key]
        if mlp_key in metrics and "NeuralNet" not in metrics:
            metrics["NeuralNet"] = metrics[mlp_key]

    add_neuralnet_alias(models_dc, metrics_dc)
    add_neuralnet_alias(models_ac, metrics_ac)
    # ------------------------------------------------------------

    stats = {
        feat: df_feat[feat].values
        for feat in FEATURE_NAMES
        if feat in df_feat.columns
    }

    return {
        "models_dc": models_dc,
        "models_ac": models_ac,
        "metrics_dc": metrics_dc,
        "metrics_ac": metrics_ac,
        "feature_stats": stats,
    }


# ======================================================================
# LOAD TASK 3 RESULTS (*.pkl)
# ======================================================================

def load_all_results(plant: str) -> Dict[str, dict]:
    """
    Loads all *_results.pkl files for a plant.

    Assumes each pickle contains a dict with:
      - "inverter_id"
      - "combined", "parallel", "loss_curves", "nn_diag"
    """
    cfg = PLANT_CONFIG.get(plant)
    if cfg is None:
        raise ValueError(f"Unknown plant '{plant}'")

    folder = cfg["results_folder"]
    pattern = os.path.join(folder, cfg["task3_results_glob"])

    files = sorted(glob.glob(pattern))
    results: Dict[str, dict] = {}

    for fpath in files:
        with open(fpath, "rb") as f:
            res = pickle.load(f)

        inv_id = res.get("inverter_id")
        if inv_id is None:
            inv_id = os.path.basename(fpath).replace("_results.pkl", "")

        results[inv_id] = res

    return results


# ======================================================================
# BUILD METRICS TABLE
# ======================================================================

def build_metrics_dataframe(results_dict: Dict[str, dict]) -> pd.DataFrame:
    """
    Convert full Task 3 results into a DataFrame for Streamlit.
    """
    rows = []

    for inv, res in results_dict.items():
        combined = res.get("combined", {})

        for side in ["dc", "ac"]:
            side_dict = combined.get(side, {})
            # side_dict expected: {model: {"rmse": float, "mae": float}}
            for model, metrics in side_dict.items():
                rows.append(
                    {
                        "inverter": inv,
                        "model": model,
                        "side": side.upper(),
                        "rmse": metrics.get("rmse", np.nan),
                        "mae": metrics.get("mae", np.nan),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["inverter", "model", "side", "rmse", "mae"])

    return pd.DataFrame(rows)


# ======================================================================
# BIAS–VARIANCE APPROX (for display only)
# ======================================================================

def compute_bias_variance(results_dict: Dict[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes proxy bias–variance from parallel RMSE values.

    Uses:
      parallel["dc_rmse"][model] → list per day
      parallel["ac_rmse"][model] → list per day
    """
    rows_dc = []
    rows_ac = []

    for inv, res in results_dict.items():
        parallel = res.get("parallel", {})

        dc_rmse = parallel.get("dc_rmse", {})
        ac_rmse = parallel.get("ac_rmse", {})

        for model, vals in dc_rmse.items():
            vals = np.asarray(vals, dtype=float)
            if vals.size:
                rows_dc.append(
                    {
                        "model": model,
                        "variance": float(np.var(vals)),
                        "bias": float(np.mean(vals)),
                    }
                )

        for model, vals in ac_rmse.items():
            vals = np.asarray(vals, dtype=float)
            if vals.size:
                rows_ac.append(
                    {
                        "model": model,
                        "variance": float(np.var(vals)),
                        "bias": float(np.mean(vals)),
                    }
                )

    return pd.DataFrame(rows_dc), pd.DataFrame(rows_ac)


# ======================================================================
# NN DIAGNOSTICS (loss curve extraction)
# ======================================================================

def extract_nn_loss_curves(results_dict: Dict[str, dict]) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Extracts NN training loss curves from Task 3 *_results.pkl files.

    Returns
    -------
    loss_dc : {inverter_id: [losses]}
    loss_ac : {inverter_id: [losses]}
    """
    dc: Dict[str, list] = {}
    ac: Dict[str, list] = {}

    for inv, res in results_dict.items():
        diag = res.get("nn_diag", {})

        if "dc" in diag and isinstance(diag["dc"], dict):
            lc = diag["dc"].get("loss_curve")
            if lc is not None:
                dc[inv] = list(lc)

        if "ac" in diag and isinstance(diag["ac"], dict):
            lc = diag["ac"].get("loss_curve")
            if lc is not None:
                ac[inv] = list(lc)

    return dc, ac


# ======================================================================
# FULL TRAINING PIPELINE — WRAPPERS AROUND YOUR EXISTING SCRIPT
# ======================================================================


def run_inverter_experiment(
    inverter_id: str,
    daily_folder: str,
    start_date_str: str,
    end_date_str: str,
    verbose: bool = True,
    save_plots: bool = False,
    plot_folder: str | None = None,
):
    """
    Train multiple regression models for one inverter over multiple days.

    - Loads all daily CSV files in `daily_folder`
    - Filters rows between start_date_str and end_date_str (inclusive)
    - Fixes missing values via interpolation + ffill/bfill
    - Trains 5 models (Linear, Ridge, Lasso, RandomForest, NeuralNet)
      on the combined dataset (all days merged)
    - Trains the same 5 models per-day (per CSV) with a train/test split
    - Computes RMSE and MAE for DC_CLEAN and AC_CLEAN
    - Trains additional NeuralNets (on combined DC & AC) for diagnostics
      (iterations, learning rate, momentum, total weights, loss curve, time)
    - Builds cost/loss curves per model:

        results["loss_curves"]["dc"][model_name] → DC cost per iteration/tree
        results["loss_curves"]["ac"][model_name] → AC cost per iteration/tree

      For:
        - Linear / Ridge / Lasso: single-point MSE (list of length 1)
        - RandomForest: per-tree MSE (growing forest)
        - NeuralNet: MLPRegressor.loss_curve_ (per-iteration loss)

    Parameters
    ----------
    inverter_id : str
        ID of the inverter (used in printouts and plot filenames).
    daily_folder : str
        Folder containing daily CSVs for this inverter.
    start_date_str : str
        e.g. "2020-05-15"
    end_date_str : str
        e.g. "2020-06-17"
    verbose : bool
        If True, print progress and metrics.
    save_plots : bool
        If True, save all plots as PNG files into plot_folder.
    plot_folder : str | None
        Destination folder for plots if save_plots is True.

    Returns
    -------
    results : dict
        {
          "inverter_id": ...,
          "combined": {
              "dc": {model: {"rmse": float, "mae": float}},
              "ac": {model: {"rmse": float, "mae": float}},
              "predictions": {
                  "ModelName_DC": {"y_true": np.array, "y_pred": np.array},
                  "ModelName_AC": {"y_true": np.array, "y_pred": np.array},
              }
          },
          "parallel": {
              "days": [list_of_date_strings_with_enough_samples],
              "dc_rmse": {model: [rmse_per_day...]},
              "ac_rmse": {model: [rmse_per_day...]},
              "dc_mae":  {model: [mae_per_day...]},
              "ac_mae":  {model: [mae_per_day...]},
              "avg_dc_rmse": {model: float},
              "avg_ac_rmse": {model: float},
              "avg_dc_mae":  {model: float},
              "avg_ac_mae":  {model: float},
          },
          "loss_curves": {
              "dc": {model: [cost_values...]},
              "ac": {model: [cost_values...]},
          },
          "nn_diag": {
              "dc": {
                  "iterations": int,
                  "learning_rate": float,
                  "momentum": float,
                  "total_weights": int,
                  "train_time": float,
                  "loss_curve": list_of_floats
              },
              "ac": {
                  "iterations": int,
                  "learning_rate": float,
                  "momentum": float,
                  "total_weights": int,
                  "train_time": float,
                  "loss_curve": list_of_floats
              }
          }
        }
    """

    # ------------------------------------------------------------------
    # 0. CONFIG
    # ------------------------------------------------------------------
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    features = [
        "IRRADIATION_CLEAN",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "DAILY_YIELD_CLEAN",
    ]
    target_dc = "DC_CLEAN"
    target_ac = "AC_CLEAN"

    if save_plots and plot_folder is not None:
        os.makedirs(plot_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. HELPER: LOAD & PREPROCESS ONE CSV
    # ------------------------------------------------------------------
    def load_and_preprocess_csv(path):
        df = pd.read_csv(path)
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

        # Restrict to date range (defensive)
        df = df[(df["DATE_TIME"] >= start_date) & (df["DATE_TIME"] <= end_date)]
        if df.empty:
            return df

        # Interpolate numerics & fill NaNs
        df = df.interpolate(method="linear")
        df = df.fillna(method="bfill").fillna(method="ffill")

        return df

    # ------------------------------------------------------------------
    # 2. LOAD ALL DAILY FILES
    # ------------------------------------------------------------------
    csv_files = sorted(glob.glob(os.path.join(daily_folder, "*.csv")))

    if verbose:
        print(f"[{inverter_id}] Found {len(csv_files)} CSV files in {daily_folder}")

    daily_dfs = []
    day_labels = []

    for f in csv_files:
        df_day = load_and_preprocess_csv(f)
        if df_day.empty:
            continue

        day_date = df_day["DATE_TIME"].dt.date.iloc[0]
        day_labels.append(str(day_date))
        daily_dfs.append(df_day)

    if not daily_dfs:
        raise ValueError(
            f"[{inverter_id}] No daily data loaded after filtering. "
            f"Check folder and date range."
        )

    # Combined dataframe (all days)
    combined_df = pd.concat(daily_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # 3. FEATURES + TARGETS (COMBINED)
    # ------------------------------------------------------------------
    X_combined = combined_df[features]
    y_combined_dc = combined_df[target_dc]
    y_combined_ac = combined_df[target_ac]

    # Single train–test split (same indices for DC & AC)
    X_train, X_test, y_train_dc, y_test_dc = train_test_split(
        X_combined, y_combined_dc, test_size=0.2, shuffle=True, random_state=42
    )
    _, _, y_train_ac, y_test_ac = train_test_split(
        X_combined, y_combined_ac, test_size=0.2, shuffle=True, random_state=42
    )

    # ------------------------------------------------------------------
    # 4. DEFINE MODELS (5 TYPES)
    # ------------------------------------------------------------------
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=10000, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "NeuralNet": MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            learning_rate_init=0.001,
            momentum=0.9,
            max_iter=2000,
            random_state=42,
        ),
    }

    # Helper: RF per-tree loss curve (MSE)
    def compute_rf_loss_curve(rf_model, X, y_true):
        """
        Compute cost per tree for a RandomForest by incrementally
        averaging trees and measuring MSE on X, y_true.
        """
        n_trees = len(rf_model.estimators_)
        if n_trees == 0:
            return []

        curves = []
        # running sum of predictions
        running_sum = None
        for i, tree in enumerate(rf_model.estimators_):
            pred_i = tree.predict(X)
            if running_sum is None:
                running_sum = pred_i
            else:
                running_sum += pred_i
            y_hat = running_sum / (i + 1)
            mse_i = mean_squared_error(y_true, y_hat)
            curves.append(mse_i)
        return curves

    # ------------------------------------------------------------------
    # 5. TRAIN + EVALUATE ON COMBINED DATA
    # ------------------------------------------------------------------
    combined_results_dc = {}      # model -> {"rmse": ..., "mae": ...}
    combined_results_ac = {}      # model -> {"rmse": ..., "mae": ...}
    combined_pred_store = {}      # "Model_DC/AC" -> {"y_true": arr, "y_pred": arr}

    # cost curves (loss_curves) per model / per target
    loss_curves_dc = {name: [] for name in models.keys()}
    loss_curves_ac = {name: [] for name in models.keys()}

    if verbose:
        print(f"\n[{inverter_id}] ================== COMBINED DATA TRAINING ==================")

    for name, model in models.items():
        # ---- DC ----
        mdl_dc = model
        mdl_dc.fit(X_train, y_train_dc)
        pred_dc = mdl_dc.predict(X_test)

        rmse_dc = np.sqrt(mean_squared_error(y_test_dc, pred_dc))
        mae_dc = mean_absolute_error(y_test_dc, pred_dc)
        mse_dc = mean_squared_error(y_test_dc, pred_dc)

        combined_results_dc[name] = {"rmse": rmse_dc, "mae": mae_dc}
        combined_pred_store[name + "_DC"] = {
            "y_true": y_test_dc.to_numpy(),
            "y_pred": pred_dc,
        }

        # default DC cost curve: single MSE value
        loss_curves_dc[name] = [mse_dc]

        # ---- AC ---- (fresh instance per target)
        if name == "Linear":
            mdl_ac = LinearRegression()
        elif name == "Ridge":
            mdl_ac = Ridge(alpha=1.0)
        elif name == "Lasso":
            mdl_ac = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
        elif name == "RandomForest":
            mdl_ac = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )
        else:  # NeuralNet
            mdl_ac = MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                learning_rate_init=0.001,
                momentum=0.9,
                max_iter=2000,
                random_state=42,
            )

        mdl_ac.fit(X_train, y_train_ac)
        pred_ac = mdl_ac.predict(X_test)

        rmse_ac = np.sqrt(mean_squared_error(y_test_ac, pred_ac))
        mae_ac = mean_absolute_error(y_test_ac, pred_ac)
        mse_ac = mean_squared_error(y_test_ac, pred_ac)

        combined_results_ac[name] = {"rmse": rmse_ac, "mae": mae_ac}
        combined_pred_store[name + "_AC"] = {
            "y_true": y_test_ac.to_numpy(),
            "y_pred": pred_ac,
        }

        # default AC cost curve: single MSE value
        loss_curves_ac[name] = [mse_ac]

        # For RandomForest, override single-point cost with per-tree curve
        if name == "RandomForest":
            loss_curves_dc[name] = compute_rf_loss_curve(mdl_dc, X_test, y_test_dc)
            loss_curves_ac[name] = compute_rf_loss_curve(mdl_ac, X_test, y_test_ac)

        if verbose:
            print(
                f"[{inverter_id}] {name:12s} | "
                f"DC  RMSE={rmse_dc:8.3f}, MAE={mae_dc:8.3f} | "
                f"AC  RMSE={rmse_ac:8.3f}, MAE={mae_ac:8.3f}"
            )

    if verbose:
        print(f"[{inverter_id}] ============================================================\n")

    # ------------------------------------------------------------------
    # 6. NEURAL NETWORK DIAGNOSTICS (COMBINED, DC & AC)
    # ------------------------------------------------------------------
    nn_diag = {"dc": {}, "ac": {}}

    # ---- DC diagnostics ----
    nn_dc = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        learning_rate_init=0.001,
        momentum=0.9,
        max_iter=2000,
        random_state=42,
    )
    start_time_dc = time.time()
    nn_dc.fit(X_train, y_train_dc)
    end_time_dc = time.time()
    training_time_dc = end_time_dc - start_time_dc
    total_weights_dc = sum(w.size for w in nn_dc.coefs_)

    nn_diag["dc"] = {
        "iterations": nn_dc.n_iter_,
        "learning_rate": nn_dc.learning_rate_init,
        "momentum": nn_dc.momentum,
        "total_weights": int(total_weights_dc),
        "train_time": float(training_time_dc),
        "loss_curve": nn_dc.loss_curve_.copy(),
    }

    # ensure NeuralNet DC loss curve uses full iterative cost
    loss_curves_dc["NeuralNet"] = list(nn_dc.loss_curve_.copy())

    # ---- AC diagnostics ----
    nn_ac = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        learning_rate_init=0.001,
        momentum=0.9,
        max_iter=2000,
        random_state=42,
    )
    start_time_ac = time.time()
    nn_ac.fit(X_train, y_train_ac)
    end_time_ac = time.time()
    training_time_ac = end_time_ac - start_time_ac
    total_weights_ac = sum(w.size for w in nn_ac.coefs_)

    nn_diag["ac"] = {
        "iterations": nn_ac.n_iter_,
        "learning_rate": nn_ac.learning_rate_init,
        "momentum": nn_ac.momentum,
        "total_weights": int(total_weights_ac),
        "train_time": float(training_time_ac),
        "loss_curve": nn_ac.loss_curve_.copy(),
    }

    # ensure NeuralNet AC loss curve uses full iterative cost
    loss_curves_ac["NeuralNet"] = list(nn_ac.loss_curve_.copy())

    if verbose:
        print(f"[{inverter_id}] ====== NEURAL NETWORK DIAGNOSTICS (COMBINED DC) ======")
        print(f"Iterations completed : {nn_diag['dc']['iterations']}")
        print(f"Learning rate (init) : {nn_diag['dc']['learning_rate']}")
        print(f"Momentum             : {nn_diag['dc']['momentum']}")
        print(f"Total weights        : {nn_diag['dc']['total_weights']}")
        print(f"Training time (sec)  : {nn_diag['dc']['train_time']:.4f}")
        print("--------------------------------------------------------------")
        print(f"[{inverter_id}] ====== NEURAL NETWORK DIAGNOSTICS (COMBINED AC) ======")
        print(f"Iterations completed : {nn_diag['ac']['iterations']}")
        print(f"Learning rate (init) : {nn_diag['ac']['learning_rate']}")
        print(f"Momentum             : {nn_diag['ac']['momentum']}")
        print(f"Total weights        : {nn_diag['ac']['total_weights']}")
        print(f"Training time (sec)  : {nn_diag['ac']['train_time']:.4f}")
        print("==============================================================\n")

    # ------------------------------------------------------------------
    # 7. PER-DAY (“PARALLEL”) TRAINING
    # ------------------------------------------------------------------
    parallel_rmse_dc = {name: [] for name in models.keys()}
    parallel_rmse_ac = {name: [] for name in models.keys()}
    parallel_mae_dc = {name: [] for name in models.keys()}
    parallel_mae_ac = {name: [] for name in models.keys()}

    valid_day_labels = []  # only days with enough samples

    if verbose:
        print(f"[{inverter_id}] =============== PER-DAY (“PARALLEL”) TRAINING ===============")

    for df_day, day_label in zip(daily_dfs, day_labels):
        # Ensure enough samples to split
        if len(df_day) < 3:
            if verbose:
                print(f"[{inverter_id}] Skipping {day_label}: not enough samples ({len(df_day)})")
            continue

        X_day = df_day[features]
        y_day_dc = df_day[target_dc]
        y_day_ac = df_day[target_ac]

        Xtr, Xte, ytr_dc, yte_dc = train_test_split(
            X_day, y_day_dc, test_size=0.2, shuffle=True, random_state=42
        )
        _, _, ytr_ac, yte_ac = train_test_split(
            X_day, y_day_ac, test_size=0.2, shuffle=True, random_state=42
        )

        valid_day_labels.append(day_label)

        for name in models.keys():
            # Fresh instance per day / target
            if name == "Linear":
                mdl_dc = LinearRegression()
                mdl_ac = LinearRegression()
            elif name == "Ridge":
                mdl_dc = Ridge(alpha=1.0)
                mdl_ac = Ridge(alpha=1.0)
            elif name == "Lasso":
                mdl_dc = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
                mdl_ac = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
            elif name == "RandomForest":
                mdl_dc = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                )
                mdl_ac = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                )
            else:  # NeuralNet
                mdl_dc = MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    momentum=0.9,
                    max_iter=2000,
                    random_state=42,
                )
                mdl_ac = MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    momentum=0.9,
                    max_iter=2000,
                    random_state=42,
                )

            # DC
            mdl_dc.fit(Xtr, ytr_dc)
            pred_dc = mdl_dc.predict(Xte)
            rmse_dc = np.sqrt(mean_squared_error(yte_dc, pred_dc))
            mae_dc = mean_absolute_error(yte_dc, pred_dc)
            parallel_rmse_dc[name].append(rmse_dc)
            parallel_mae_dc[name].append(mae_dc)

            # AC
            mdl_ac.fit(Xtr, ytr_ac)
            pred_ac = mdl_ac.predict(Xte)
            rmse_ac = np.sqrt(mean_squared_error(yte_ac, pred_ac))
            mae_ac = mean_absolute_error(yte_ac, pred_ac)
            parallel_rmse_ac[name].append(rmse_ac)
            parallel_mae_ac[name].append(mae_ac)

    # Average per-day metrics
    avg_parallel_rmse_dc = {
        name: float(np.mean(vals)) for name, vals in parallel_rmse_dc.items() if len(vals) > 0
    }
    avg_parallel_rmse_ac = {
        name: float(np.mean(vals)) for name, vals in parallel_rmse_ac.items() if len(vals) > 0
    }
    avg_parallel_mae_dc = {
        name: float(np.mean(vals)) for name, vals in parallel_mae_dc.items() if len(vals) > 0
    }
    avg_parallel_mae_ac = {
        name: float(np.mean(vals)) for name, vals in parallel_mae_ac.items() if len(vals) > 0
    }

    if verbose:
        print(f"\n[{inverter_id}] ===== AVERAGE PER-DAY (“PARALLEL”) RESULTS =====")
        for name in models.keys():
            if name in avg_parallel_rmse_dc:
                print(
                    f"{name:12s} | DC  RMSE={avg_parallel_rmse_dc[name]:8.3f}, "
                    f"MAE={avg_parallel_mae_dc[name]:8.3f} | "
                    f"AC  RMSE={avg_parallel_rmse_ac[name]:8.3f}, "
                    f"MAE={avg_parallel_mae_ac[name]:8.3f}"
                )
        print("===================================================================\n")

    # ------------------------------------------------------------------
    # 8. PACK RESULTS INTO A SINGLE DICTIONARY
    # ------------------------------------------------------------------
    results = {
        "inverter_id": inverter_id,
        "combined": {
            "dc": combined_results_dc,
            "ac": combined_results_ac,
            "predictions": combined_pred_store,
        },
        "parallel": {
            "days": valid_day_labels,
            "dc_rmse": parallel_rmse_dc,
            "ac_rmse": parallel_rmse_ac,
            "dc_mae": parallel_mae_dc,
            "ac_mae": parallel_mae_ac,
            "avg_dc_rmse": avg_parallel_rmse_dc,
            "avg_ac_rmse": avg_parallel_rmse_ac,
            "avg_dc_mae": avg_parallel_mae_dc,
            "avg_ac_mae": avg_parallel_mae_ac,
        },
        "loss_curves": {
            "dc": loss_curves_dc,
            "ac": loss_curves_ac,
        },
        "nn_diag": nn_diag,
    }

    # ------------------------------------------------------------------
    # 9. SAVE PLOTS (NO plt.show())
    # ------------------------------------------------------------------
    if save_plots and plot_folder is not None:
        # ---------- A. Combined predictions: Actual vs Predicted + Residuals ----------
        for key, vals in combined_pred_store.items():
            y_true = vals["y_true"]
            y_pred = vals["y_pred"]

            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"{inverter_id} — {key} — Combined Model")

            # Scatter: Actual vs Predicted
            ax[0].scatter(y_true, y_pred, alpha=0.7)
            mn = min(y_true.min(), y_pred.min())
            mx = max(y_true.max(), y_pred.max())
            ax[0].plot([mn, mx], [mn, mx], "r--")
            ax[0].set_title("Actual vs Predicted")
            ax[0].set_xlabel("Actual")
            ax[0].set_ylabel("Predicted")
            ax[0].grid(True)

            # Residuals vs Predicted
            residuals = y_true - y_pred
            ax[1].scatter(y_pred, residuals, alpha=0.6)
            ax[1].axhline(0, color="red", linestyle="--")
            ax[1].set_title("Residuals vs Predicted")
            ax[1].set_xlabel("Predicted")
            ax[1].set_ylabel("Residual")
            ax[1].grid(True)

            # Residual distribution
            ax[2].hist(residuals, bins=20, alpha=0.8)
            ax[2].set_title("Residual Distribution")
            ax[2].set_xlabel("Residual")
            ax[2].set_ylabel("Frequency")
            ax[2].grid(True)

            fname = f"{inverter_id}_combined_{key}_performance.png"
            fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ---------- B. RMSE bar: Combined DC vs AC ----------
        labels = list(models.keys())
        rmse_dc_vals = [combined_results_dc[m]["rmse"] for m in labels]
        rmse_ac_vals = [combined_results_ac[m]["rmse"] for m in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, rmse_dc_vals, width, label="DC RMSE")
        ax.bar(x + width / 2, rmse_ac_vals, width, label="AC RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE")
        ax.set_title(f"{inverter_id} — Combined Model RMSE (DC vs AC)")
        ax.legend()
        ax.grid(axis="y")

        fname = f"{inverter_id}_combined_rmse_dc_vs_ac.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- C. Neural Network Loss Curve (DC) ----------
        fig, ax = plt.subplots(figsize=(10, 5))
        loss_arr_dc = np.array(nn_diag["dc"]["loss_curve"])
        ax.plot(loss_arr_dc, label="Training Loss (DC)")

        ax.set_title(f"{inverter_id} — Neural Network Loss Curve (DC, Combined)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

        fname = f"{inverter_id}_nn_loss_curve_DC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- C2. Neural Network Loss Curve (AC) ----------
        fig, ax = plt.subplots(figsize=(10, 5))
        loss_arr_ac = np.array(nn_diag["ac"]["loss_curve"])
        ax.plot(loss_arr_ac, label="Training Loss (AC)")

        ax.set_title(f"{inverter_id} — Neural Network Loss Curve (AC, Combined)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

        fname = f"{inverter_id}_nn_loss_curve_AC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- D. NN diagnostic bars (use DC stats) ----------
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.ravel()

        axs[0].bar(["Iterations (DC)"], [nn_diag["dc"]["iterations"]])
        axs[0].set_title("Training Iterations (DC)")
        axs[0].grid(axis="y")

        axs[1].bar(["Learning Rate (DC)"], [nn_diag["dc"]["learning_rate"]])
        axs[1].set_title("Learning Rate (DC)")
        axs[1].grid(axis="y")

        axs[2].bar(["Total Weights (DC)"], [nn_diag["dc"]["total_weights"]])
        axs[2].set_title("Model Size (Total Weights, DC)")
        axs[2].grid(axis="y")

        axs[3].bar(["Training Time (s, DC)"], [nn_diag["dc"]["train_time"]])
        axs[3].set_title("Training Time (DC)")
        axs[3].grid(axis="y")

        fig.suptitle(f"{inverter_id} — Neural Network Diagnostics (DC)", y=1.02)
        fig.tight_layout()

        fname = f"{inverter_id}_nn_diagnostics_DC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- E. Combined vs Average Parallel metrics (RMSE / MAE) ----------
        def plot_combined_vs_parallel(
            metric_combined_dc,
            metric_combined_ac,
            metric_parallel_dc,
            metric_parallel_ac,
            title_suffix,
            suffix_file,
        ):
            labels_loc = list(models.keys())
            x_loc = np.arange(len(labels_loc))
            width_loc = 0.18

            fig_loc, ax_loc = plt.subplots(figsize=(12, 6))

            c_dc = [metric_combined_dc[m] for m in labels_loc]
            c_ac = [metric_combined_ac[m] for m in labels_loc]
            p_dc = [metric_parallel_dc.get(m, np.nan) for m in labels_loc]
            p_ac = [metric_parallel_ac.get(m, np.nan) for m in labels_loc]

            ax_loc.bar(x_loc - 1.5 * width_loc, c_dc, width_loc, label="Combined DC")
            ax_loc.bar(x_loc - 0.5 * width_loc, c_ac, width_loc, label="Combined AC")
            ax_loc.bar(x_loc + 0.5 * width_loc, p_dc, width_loc, label="Avg Parallel DC")
            ax_loc.bar(x_loc + 1.5 * width_loc, p_ac, width_loc, label="Avg Parallel AC")

            ax_loc.set_xticks(x_loc)
            ax_loc.set_xticklabels(labels_loc)
            ax_loc.set_ylabel(title_suffix)
            ax_loc.set_title(f"{inverter_id} — Combined vs Average Parallel ({title_suffix})")
            ax_loc.legend()
            ax_loc.grid(axis="y")

            fig_loc.tight_layout()
            fig_loc.savefig(os.path.join(plot_folder, suffix_file),
                            dpi=150, bbox_inches="tight")
            plt.close(fig_loc)

        # RMSE comparison
        plot_combined_vs_parallel(
            {m: combined_results_dc[m]["rmse"] for m in models.keys()},
            {m: combined_results_ac[m]["rmse"] for m in models.keys()},
            avg_parallel_rmse_dc,
            avg_parallel_rmse_ac,
            "RMSE",
            f"{inverter_id}_combined_vs_parallel_RMSE.png",
        )

        # MAE comparison
        plot_combined_vs_parallel(
            {m: combined_results_dc[m]["mae"] for m in models.keys()},
            {m: combined_results_ac[m]["mae"] for m in models.keys()},
            avg_parallel_mae_dc,
            avg_parallel_mae_ac,
            "MAE",
            f"{inverter_id}_combined_vs_parallel_MAE.png",
        )

        # ---------- F. Per-day (“parallel”) RMSE over time ----------
        days_idx = np.arange(len(valid_day_labels))

        # DC
        fig, ax = plt.subplots(figsize=(14, 6))
        for name in models.keys():
            if len(parallel_rmse_dc[name]) == len(valid_day_labels):
                ax.plot(days_idx, parallel_rmse_dc[name], marker="o", label=name)
        ax.set_xticks(days_idx)
        ax.set_xticklabels(valid_day_labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE (DC)")
        ax.set_xlabel("Day")
        ax.set_title(f"{inverter_id} — Per-Day RMSE (DC, Parallel Training)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        fname = f"{inverter_id}_per_day_rmse_DC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # AC
        fig, ax = plt.subplots(figsize=(14, 6))
        for name in models.keys():
            if len(parallel_rmse_ac[name]) == len(valid_day_labels):
                ax.plot(days_idx, parallel_rmse_ac[name], marker="o", label=name)
        ax.set_xticks(days_idx)
        ax.set_xticklabels(valid_day_labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE (AC)")
        ax.set_xlabel("Day")
        ax.set_title(f"{inverter_id} — Per-Day RMSE (AC, Parallel Training)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        fname = f"{inverter_id}_per_day_rmse_AC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 10. RETURN RESULTS (your main script pickles per inverter)
    # ------------------------------------------------------------------
    return results


def train_single_inverter(plant: str, inverter_id: str, verbose: bool = False) -> dict:
    """
    Calls your existing Task3_FullTraining.run_inverter_experiment() script
    for this inverter, and returns its results dictionary.

    This is used only in the "Retrain Models" tab.
    """

    cfg = PLANT_CONFIG[plant]

    # Daily CSVs should live in: .../00 Excel clean file/Plant X/Daily_Inverter_Data/<inverter_id>/*.csv
    base_folder = os.path.join(cfg["clean_root"], "Daily_Inverter_Data")
    inv_folder = os.path.join(base_folder, inverter_id)

    results = run_inverter_experiment(
        inverter_id=inverter_id,
        daily_folder=inv_folder,
        start_date_str="2020-05-15",
        end_date_str="2020-06-17",
        verbose=verbose,
        save_plots=True,
        plot_folder=os.path.join(cfg["results_folder"], inverter_id),
    )

    return results


def train_all_inverters(plant: str, progress_callback=None, verbose: bool = False) -> None:
    """
    Train all inverters for a given plant via your full pipeline.
    """
    inverters = get_inverters_for_plant(plant)
    total = len(inverters)

    for i, inv in enumerate(inverters, start=1):
        if progress_callback is not None:
            progress_callback(i, total)
        train_single_inverter(plant, inv, verbose=verbose)

# -------------------------
# DASHBOARD 2
# -------------------------
def predict_label(svm, X_input, thr: float):
    score = float(svm.decision_function(X_input))
    label = 1 if score >= thr else 0
    return score, label
