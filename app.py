import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from utils import predict_label
import nbformat
import joblib
import os
import glob
import pickle
import shutil
import base64

# --------------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Solar SVM Dashboard", layout="wide")
st.markdown(
    """
    <style>
    html,body,[class*='css']{font-family:'Inter',sans-serif}
    h1,h2,h3,h4{font-weight:700!important}
    .stButton>button{background:#00c2ff!important;color:black!important;border-radius:8px;border:0;padding:.6rem 1.2rem;font-weight:600}
    .stButton>button:hover{background:#84e1ff!important}
    </style>
    """,
    unsafe_allow_html=True,)

# -------------------------
# DASHBOARD 1
# -------------------------

from config import (
    PLANT_CONFIG,
    FEATURES_CONFIG,
    FEATURE_NAMES,
    MODEL_NAMES,
    TASK4_FOLDER,
    TASK4_RESULTS_GLOB,
    TASK5_RF_AC_DC,
    TASK5_RF_SIMPLE,
    TASK6_FOLDER,
    TASK6_LSTM_PKL,
)

from utils import (
    get_inverters_for_plant,
    get_trained_models_for_inverter,
    make_feature_input_defaults,
    compute_feature_importance,
    load_all_results,
    build_metrics_dataframe,
    compute_bias_variance,
    extract_nn_loss_curves,
    train_single_inverter,
    train_all_inverters,
)

# --------------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------------------

def sidebar_controls():
    st.sidebar.title("Controls")

    plant = st.sidebar.selectbox("Plant", list(PLANT_CONFIG.keys()), index=0)
    inverters = get_inverters_for_plant(plant)
    inverter_id = st.sidebar.selectbox("Inverter", inverters, index=0)

    target_side = st.sidebar.radio(
        "Target to predict",
        ["DC power (DC_CLEAN)", "AC power (AC_CLEAN)"],
        index=0,
    )
    target_key = "dc" if target_side.startswith("DC") else "ac"

    # Primary models: exclude NeuralNet (fast retrain may not support it)
    primary_options = [m for m in MODEL_NAMES if m != "NeuralNet"]
    primary_model = st.sidebar.selectbox("Primary model", primary_options, index=0)

    # Compare models: include ALL, including NeuralNet
    compare_models = st.sidebar.multiselect(
        "Models to compare",
        options=MODEL_NAMES,
        default=MODEL_NAMES,
    )

    return plant, inverter_id, target_key, primary_model, compare_models



def sidebar_feature_inputs(feature_ui_cfg):
    st.sidebar.header("Environmental conditions")

    user_values = {}
    for feat in FEATURES_CONFIG:
        name = feat["name"]
        label = feat["label"]
        cfg = feature_ui_cfg.get(name)
        if not cfg:
            continue
        user_values[name] = st.sidebar.slider(
            label,
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]) if cfg["step"] > 0 else 0.1,
        )
    return user_values


# --------------------------------------------------------------------------------------
# PAGE 1 – INTERACTIVE PREDICTIONS (FAST RETRAIN)
# --------------------------------------------------------------------------------------

def page_predictions(
    plant: str,
    inverter_id: str,
    target_key: str,
    primary_model_name: str,
    compare_models: list[str],
):
    st.header("🔮 Interactive Predictions")

    # 1) Train lightweight models for this inverter
    try:
        with st.spinner("Training lightweight models for selected inverter..."):
            train_result = get_trained_models_for_inverter(plant, inverter_id)
    except Exception as e:
        st.error(f"Could not train fast models for {plant} – {inverter_id}.\n\n{e}")
        return

    models_dc = train_result["models_dc"]
    models_ac = train_result["models_ac"]
    metrics_dc = train_result["metrics_dc"]
    metrics_ac = train_result["metrics_ac"]
    feature_stats = train_result["feature_stats"]

    # 2) Sidebar sliders from feature stats
    feature_ui_cfg = make_feature_input_defaults(feature_stats)
    user_inputs = sidebar_feature_inputs(feature_ui_cfg)

    if not user_inputs:
        st.error("No feature inputs available.")
        return

    X_user = np.array([[user_inputs[f] for f in FEATURE_NAMES]], dtype=float)

    if target_key == "dc":
        models_for_target = models_dc
        metrics_for_target = metrics_dc
        target_label = "DC_CLEAN"
    else:
        models_for_target = models_ac
        metrics_for_target = metrics_ac
        target_label = "AC_CLEAN"

    st.markdown(
        f"**Plant:** `{plant}` · **Inverter:** `{inverter_id}` · **Target:** `{target_label}`"
    )

    st.markdown("---")
    st.subheader("Primary model prediction")

    if primary_model_name not in models_for_target:
        st.error(f"Primary model `{primary_model_name}` not available.")
        st.write("Available models:", ", ".join(sorted(models_for_target.keys())))
        return

    primary_model = models_for_target[primary_model_name]
    y_pred_primary = float(primary_model.predict(X_user)[0])

    col_pred, col_info = st.columns([2, 1])
    with col_pred:
        st.metric(
            label=f"{target_label} — {primary_model_name}",
            value=f"{y_pred_primary:,.3f}",
        )
        st.caption("Current input values")
        st.write(pd.DataFrame([user_inputs]))
    with col_info:
        st.markdown("**Test metrics (fast retrain)**")
        m = metrics_for_target.get(primary_model_name, {})
        if m:
            st.write(f"- RMSE: `{m['rmse']:.4f}`")
            st.write(f"- MAE: `{m['mae']:.4f}`")
        else:
            st.write("No metrics available.")

    # 3) Comparison plot across models
    st.markdown("---")
    st.subheader("Model comparison for current conditions")

    compare_models = compare_models or [primary_model_name]              # <---------------------------------------

    available = set(models_for_target.keys())
    missing = [m for m in compare_models if m not in available]

    if missing:
        st.info(
            "These models are only available in the full Task 3 training "
            "(no fast retrain for this tab): "
            + ", ".join(missing)
        )

    rows = []
    for name in compare_models:
        if name not in available:
            continue
        model = models_for_target[name]
        y_hat = float(model.predict(X_user)[0])
        row = {"Model": name, "Prediction": y_hat}
        m = metrics_for_target.get(name, {})
        row["RMSE"] = m.get("rmse", np.nan)
        row["MAE"] = m.get("mae", np.nan)
        rows.append(row)                                                 # <---------------------------------------

    if rows:
        df_compare = pd.DataFrame(rows)
        col1, col2 = st.columns(2)

        with col1:
            fig_pred = px.bar(
                df_compare,
                x="Model",
                y="Prediction",
                text="Prediction",
                title=f"Predicted {target_label} by model",
            )
            fig_pred.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_pred.update_layout(yaxis_title="Prediction")
            st.plotly_chart(fig_pred, use_container_width=True)

        with col2:
            fig_rmse = px.bar(
                df_compare,
                x="Model",
                y="RMSE",
                text="RMSE",
                title=f"Test RMSE (fast retrain) — {target_label}",
            )
            fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_rmse.update_layout(yaxis_title="RMSE")
            st.plotly_chart(fig_rmse, use_container_width=True)

        st.write("Detailed table")
        st.dataframe(df_compare.set_index("Model"))
    else:
        st.info("No models selected for comparison.")

    # 4) Feature importance for primary model
    st.markdown("---")
    st.subheader("📊 Feature importance / coefficients")

    importance = compute_feature_importance(primary_model, FEATURE_NAMES)
    if not importance:
        st.info("Feature importances not available for this model.")
    else:
        df_imp = (
            pd.DataFrame(
                {
                    "Feature": list(importance.keys()),
                    "Importance": list(importance.values()),
                }
            )
            .sort_values("Importance", ascending=False)
        )

        fig_imp = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Feature importance / |coefficients| — {primary_model_name}",
        )
        fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.write("Numeric values")
        st.dataframe(df_imp.set_index("Feature"))

    st.caption(
        "Note: This tab retrains simpler models quickly for interactive predictions. "
        "The full Task 3 training (with parallel metrics, NN diagnostics, etc.) is "
        "visualised in the other tabs from your pre-computed *_results.pkl files."
    )


# --------------------------------------------------------------------------------------
# PAGE 2 – TASK 3 RESULTS (FROM *_results.pkl)
# --------------------------------------------------------------------------------------

def page_task3_results(plant: str, compare_models: list[str]):    # <------------------------------------------------
    st.header("📊 Task 3 — Full Inverter Results (from *_results.pkl)")

    try:
        results = load_all_results(plant)
    except Exception as e:
        st.error(f"Error loading Task 3 results for {plant}:\n\n{e}")
        return

    if not results:
        st.warning("No *_results.pkl files found for this plant. Retrain first.")
        return

    df_metrics = build_metrics_dataframe(results) # <======================================================================

    # Only keep selected models, if any selection given
    if compare_models:
        allowed_models = set(compare_models)

        # Map UI names → names used in Task 3 results
        if "LinearRegression" in allowed_models:
            allowed_models.add("Linear")  # Task 3 uses "Linear"

        # (add any other mappings here if needed)

        df_metrics = df_metrics[df_metrics["model"].isin(allowed_models)] # <======================================================================


    st.subheader("Combined RMSE/MAE per model")

    inv_options = ["All inverters"] + sorted(df_metrics["inverter"].unique())
    inv_choice = st.selectbox("Scope", inv_options, index=0)

    if inv_choice == "All inverters":
        df_summary = (
            df_metrics.groupby(["model", "side"])
            .agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
            .reset_index()
        )
        st.write("Average across all inverters")
    else:
        df_summary = df_metrics[df_metrics["inverter"] == inv_choice].copy()
        st.write(f"Metrics for inverter `{inv_choice}`")

    st.dataframe(df_summary)

    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = px.bar(
            df_summary,
            x="model",
            y="rmse",
            color="side",
            barmode="group",
            title="RMSE by model & side",
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col2:
        fig_mae = px.bar(
            df_summary,
            x="model",
            y="mae",
            color="side",
            barmode="group",
            title="MAE by model & side",
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # Bias–variance
    st.markdown("---")
    st.subheader("Bias–Variance proxies (from parallel per-day training)")

    df_dc, df_ac = compute_bias_variance(results) # <======================================================

    if compare_models:
        allowed_models = set(compare_models)
        if "LinearRegression" in allowed_models:
            allowed_models.add("Linear")

        df_dc = df_dc[df_dc["model"].isin(allowed_models)]
        df_ac = df_ac[df_ac["model"].isin(allowed_models)]  # <======================================================



    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**DC side**")
        if df_dc.empty:
            st.info("No DC parallel metrics found.")
        else:
            st.dataframe(df_dc.set_index("model"))
            fig_dc = px.scatter(
                df_dc,
                x="variance",
                y="bias",
                text="model",
                title="Bias–variance proxy — DC",
            )
            fig_dc.update_traces(textposition="top center")
            fig_dc.update_layout(xaxis_title="Variance proxy", yaxis_title="Bias proxy")
            st.plotly_chart(fig_dc, use_container_width=True)

    with col4:
        st.markdown("**AC side**")
        if df_ac.empty:
            st.info("No AC parallel metrics found.")
        else:
            st.dataframe(df_ac.set_index("model"))
            fig_ac = px.scatter(
                df_ac,
                x="variance",
                y="bias",
                text="model",
                title="Bias–variance proxy — AC",
            )
            fig_ac.update_traces(textposition="top center")
            fig_ac.update_layout(xaxis_title="Variance proxy", yaxis_title="Bias proxy")
            st.plotly_chart(fig_ac, use_container_width=True)

    # NN loss curves
    st.markdown("---")   # <----------------------------------------------------------------------
    st.subheader("Neural Network training loss (mean DC vs AC)")

    # Optional: only show if NeuralNet is selected
    NN_NAME = "NeuralNet"
    if compare_models and NN_NAME not in compare_models:
        st.info(
            f"Include `{NN_NAME}` in **Models to compare** to view the NN loss curves."
        )
        return

    loss_dc, loss_ac = extract_nn_loss_curves(results)

    if not loss_dc and not loss_ac:
        st.info("No NN loss curves found in nn_diag.")
        return         # <----------------------------------------------------------------------


    curves_dc = list(loss_dc.values())
    curves_ac = list(loss_ac.values())

    max_len_dc = max(len(c) for c in curves_dc) if curves_dc else 0
    max_len_ac = max(len(c) for c in curves_ac) if curves_ac else 0
    L = min(max_len_dc, max_len_ac) if max_len_dc and max_len_ac else max(
        max_len_dc, max_len_ac
    )

    mean_dc = None
    mean_ac = None

    if curves_dc:
        mat_dc = np.full((len(curves_dc), max_len_dc), np.nan)
        for i, c in enumerate(curves_dc):
            mat_dc[i, : len(c)] = c
        mean_dc = np.nanmean(mat_dc, axis=0)[:L]

    if curves_ac:
        mat_ac = np.full((len(curves_ac), max_len_ac), np.nan)
        for i, c in enumerate(curves_ac):
            mat_ac[i, : len(c)] = c
        mean_ac = np.nanmean(mat_ac, axis=0)[:L]

    epochs = np.arange(L)
    data_loss = []
    if mean_dc is not None:
        data_loss.append(pd.DataFrame({"epoch": epochs, "loss": mean_dc, "side": "DC"}))
    if mean_ac is not None:
        data_loss.append(pd.DataFrame({"epoch": epochs, "loss": mean_ac, "side": "AC"}))


####################################################################################################
    if data_loss:
        df_loss = pd.concat(data_loss, ignore_index=True)

        # No Streamlit slider anymore – always use full range of epochs
        max_epoch = int(df_loss["epoch"].max())
        df_loss_plot = df_loss.copy()

        if df_loss_plot.empty:
            st.info("No NN loss curves found in nn_diag.")
            return

        sides = df_loss_plot["side"].unique()
        start_epoch = int(df_loss_plot["epoch"].min())

        # 1) Base figure + initial traces
        fig_loss = go.Figure()
        for side in sides:
            df0 = df_loss_plot[
                (df_loss_plot["side"] == side)
                & (df_loss_plot["epoch"] <= start_epoch)
            ]
            fig_loss.add_trace(
                go.Scatter(
                    x=df0["epoch"],
                    y=df0["loss"],
                    mode="lines",
                    name=side,
                )
            )

        # Fixed y-axis for stability
        y_max = float(df_loss_plot["loss"].max()) * 1.05

        fig_loss.update_layout(
            title="Mean NN training loss — DC vs AC",
            xaxis_title="epoch",
            yaxis_title="loss",
            xaxis=dict(range=[0, start_epoch], autorange=False),
            yaxis=dict(range=[0, y_max], autorange=False),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=1.0,
                    y=1.15,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 40, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        )

        # 2) Frames: extend lines and x-axis up to each epoch
        frames = []
        for e in range(start_epoch + 1, max_epoch + 1):
            frame_data = []
            for side in sides:
                dfs = df_loss_plot[
                    (df_loss_plot["side"] == side)
                    & (df_loss_plot["epoch"] <= e)
                ]
                frame_data.append(
                    go.Scatter(
                        x=dfs["epoch"],
                        y=dfs["loss"],
                        mode="lines",
                        name=side,
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(e),
                    layout=go.Layout(
                        xaxis=dict(range=[0, e], autorange=False),
                        yaxis=dict(range=[0, y_max], autorange=False),
                    ),
                )
            )

        fig_loss.frames = frames

        # 3) White Plotly slider at the BOTTOM of the plot
        steps = []
        for e in range(start_epoch, max_epoch + 1):
            steps.append(
                dict(
                    method="animate",
                    args=[
                        [str(e)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                    label=str(e),
                )
            )

        sliders = [
            dict(
                active=len(steps) - 1,
                currentvalue={"prefix": "Epoch: "},
                pad={"t": 0, "b": 100},
                x=0.0,
                xanchor="left",
                len=1.0,
                y=0,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.05)",   # 20% opacity white
                # bordercolor="rgba(255, 255, 255, 0.2)",  # optional
                steps=steps,
            )
        ]

        fig_loss.update_layout(
            sliders=sliders,
            margin=dict(t=80, b=160),
        )

        st.plotly_chart(fig_loss, use_container_width=True)


def dashboard_1():
    st.title('Regression Model')

    plant, inverter_id, target_key, primary_model, compare_models = sidebar_controls()

    tabs = st.tabs(["Predictions","Task 3 Results",])

    with tabs[0]:
        page_predictions(plant, inverter_id, target_key, primary_model, compare_models)

    with tabs[1]:                                        
        page_task3_results(plant, compare_models)

    def dashboard_1():
        st.title("Regression Model")
        
        plant, inverter_id, target_key, primary_model, compare_models = sidebar_controls()

        tabs = st.tabs(["Predictions", "Task 3 Results",])

        with tabs[0]:
            page_predictions(plant, inverter_id, target_key, primary_model, compare_models)

        with tabs[1]:
            page_task3_results(plant)


# -------------------------
# DASHBOARD 2
# -------------------------

# -------------------------
# Plot helpers (kept in app for UI control)
# -------------------------
def apply_plot_style(fig, axis_title_size=16, tick_size=16, legend_size=16):
    fig.update_layout(
        xaxis_title_font={"size": axis_title_size},
        yaxis_title_font={"size": axis_title_size},
        legend={"font": {"size": legend_size}},
    )
    fig.update_xaxes(tickfont={"size": tick_size})
    fig.update_yaxes(tickfont={"size": tick_size})
    return fig

def draw_ale_mini_plot(ale_df, feature, current_val):
    df = ale_df[ale_df["feature"] == feature]
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["x"],y=df["ale"],mode="lines",line=dict(color="blue", width=2),showlegend=False,))
    i = (df["x"] - current_val).abs().idxmin()
    fig.add_trace(go.Scatter(x=[df.at[i, "x"]], y=[df.at[i, "ale"]],mode="markers",marker=dict(color="white", size=10),showlegend=False,) )
    fig.update_layout(height=120, margin=dict(l=20, r=20, t=10, b=10))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title=None, nticks=5, showgrid=True, tickfont=dict(size=12, color="white"), tickformat=".2f")

    return fig

def make_cm_fig(cm):
    fig = px.imshow(cm,text_auto=True,color_continuous_scale="Blues",x=["Predict Optimal", "Predict Suboptimal"],y=["Actual Optimal", "Actual Suboptimal"],)
    fig.add_shape(type="line", x0=-0.5, y0=0.5, x1=1.5, y1=0.5, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0.5, y0=-0.5, x1=0.5, y1=1.5, line=dict(color="black", width=2))
    fig.update_layout(font=dict(size=16))
    return apply_plot_style(fig, axis_title_size=14)


def make_hist(scores, labels, thr, axis_title_size=18, font_size=16, nbins=50):
    df = pd.DataFrame({"score": scores,"Label Name": pd.Series(labels).map({0: "Optimal", 1: "Suboptimal"}),})
    fig = px.histogram(df, x="score", color="Label Name", nbins=nbins, opacity=0.6, barmode="overlay")
    fig.add_vline(x=thr, line_color="white", line_dash="dash")
    fig.update_layout(font=dict(size=font_size))
    return apply_plot_style(fig, axis_title_size=axis_title_size)

# -------------------------
# Load models + results (cached)
# -------------------------
@st.cache_resource
def load_all_models():
    data = {}

    # Folders containing exported outputs
    result_files = glob.glob("results/*.csv")
    model_files  = glob.glob("models/*.pkl")

    # Extract unique (plant, version, inverter_id)
    inverter_keys = set()

    for path in model_files:
        # example: models/svm_Plant1_v1_1BY6WEcLGh8j5v7.pkl
        base = os.path.basename(path).replace(".pkl", "")
        _, plant, version, inverter = base.split("_", 3)
        inverter_keys.add((plant, version, inverter))

    for plant, version, inverter in inverter_keys:
        key = f"{plant}_{version}_{inverter}"

    # ----------------------------------------------------
    # 2. Load artifacts for each inverter
    # ----------------------------------------------------
    for plant, version, inverter in sorted(inverter_keys):
        key = f"{plant}_{version}_{inverter}"

        data[key] = {
            # Load model EXACTLY from model_files (no double .pkl)
            "svm": joblib.load(f"models/svm_{plant}_{version}_{inverter}.pkl"),

            # Result artifacts
            "cm_svm": np.load(f"results/confusion_{plant}_{version}_{inverter}.npy"),
            "metrics": pd.read_csv(f"results/metrics_{plant}_{version}_{inverter}.csv"),
            "features": np.load(f"results/features_{plant}_{version}_{inverter}.npy", allow_pickle=True),
            "threshold": float(
                pd.read_csv(f"results/thresholds_{plant}_{version}_{inverter}.csv")["threshold"].iloc[0]
            ),
            "X_te": np.load(f"results/X_te_{plant}_{version}_{inverter}.npy"),
            "y_te": np.load(f"results/y_te_{plant}_{version}_{inverter}.npy"),
            "drop_importance": pd.read_csv(f"results/drop_importance_{plant}_{version}_{inverter}.csv"),
            "ale": pd.read_csv(f"results/ale_{plant}_{version}_{inverter}.csv"),
        }

    return data


artifacts = load_all_models()

# -------------------------
# Page Setup
# -------------------------
def dashboard_2():
    st.title("Classification Model")

    # -------------------------
    # Sidebar Controls
    # -------------------------
    st.sidebar.header("Controls")

    # ---- Plant selector ----
    plant = st.sidebar.selectbox("Plant:", ["Plant1", "Plant2"])

    # ---- Version selector ----
    version = st.sidebar.selectbox("Version:", ["v1", "v2"])

    # ---- List available inverters for this plant+version ----
    available_inverters = sorted([
        key.rsplit("_", 1)[-1]  
        for key in artifacts.keys()
        if key.startswith(f"{plant}_{version}")
    ])

    if not available_inverters:
        st.sidebar.error(f"No inverters found for {plant} {version}")
        st.stop()

    # ---- Select inverter ----
    inverter = st.sidebar.selectbox("Inverter:", available_inverters)

    # ---- Compose key ----
    idkey = f"{plant}_{version}_{inverter}"
    m = artifacts[idkey]

    # Load objects
    svm = m["svm"]
    features = m["features"]
    thr = m["threshold"]
    ale_df = m["ale"]

    st.sidebar.write(f"**Threshold:** `{thr:.4f}`")

    # ---------------------------------------------------------
    # Prediction Section
    # ---------------------------------------------------------
    st.markdown("---")
    st.header("🤖 PREDICTION")
    st.write("### ▫️ Base Features")
    st.write(" ")
    
    if plant == 'Plant1':
        i =14000.0
    elif plant == 'Plant2':
        i = 1400.0
        
    slider_ranges = {
        "AC_CLEAN": (0.0, 1400.0), "DC_CLEAN": (0.0, i),
        "IRRADIATION_CLEAN": (0.0, 1.20), "DAILY_YIELD_CLEAN": (0.0, 10000.0),
        "TOTAL_YIELD_CLEAN": (0, 2500000000), "AMBIENT_TEMPERATURE": (0.0, 40.0),
        "MODULE_TEMPERATURE": (0.0, 70.0),}

    cols = st.columns(3)
    inputs = {}
    base_features = [f for f in features if f not in ["DC/IRRA", "AC/IRRA"]]
    show_ac_slider = (plant == "Plant2" and version == "v2")

    for i, feat in enumerate(base_features):
            with cols[i % 3]:
                # st.markdown(f"##### {feat}")
                st.markdown(f"<div style='text-align: center; font-size: 16px; font-weight: 600;'>{feat}</div>", unsafe_allow_html=True)

                # ----- SLIDER -----
                lo, hi = slider_ranges.get(feat, (0.0, 2000.0))
                slider_key = f"slider_{feat}"

                val = st.slider( "",float(lo), float(hi),float((lo + hi) / 2), key=slider_key,)
                inputs[feat] = val

                fig_key = f"ale_fig_{feat}"

                fig_key = f"ale_fig_{version}_{plant}_{feat}"
                prev_key = f"{fig_key}_prev"

                # Read old slider value
                prev_val = st.session_state.get(prev_key, None)

                # Always recompute ALE plot for correct model/state
                fig = draw_ale_mini_plot(ale_df, feat, val)
                st.plotly_chart(fig, use_container_width=True)
                
    if show_ac_slider:
        with cols[0]:
            st.markdown("### Additional Features")
            st.markdown("")
            st.markdown("<div style='text-align:center; font-size:16px; font-weight:600;'>AC_CLEAN</div>", unsafe_allow_html=True)
            inputs["AC_CLEAN"] = st.slider("", 0.0, 1400.0, 700.0, key="slider_AC_CLEAN")
            
    elif "AC_CLEAN" not in base_features:
        inputs["AC_CLEAN"] = 0.0

    irr = inputs.get("IRRADIATION_CLEAN", 0.0)
    inputs["DC/IRRA"] = inputs.get("DC_CLEAN", 0.0) / (irr + 1e-6)
    inputs["AC/IRRA"] = inputs.get("AC_CLEAN", 0.0) / (irr + 1e-6)

    if not show_ac_slider:
        st.write("### ▫️ Additional Features")
        st.write(" ")
    # -- First row: numeric values
    if not show_ac_slider: 
        num_cols = st.columns(3)
        with num_cols[0]:
            st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:600;'>DC/IRRA: {inputs['DC/IRRA']:.4f}</div>",unsafe_allow_html=True)
        with num_cols[1]:
            st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:600;'>AC/IRRA: {inputs['AC/IRRA']:.4f}</div>",unsafe_allow_html=True)
        with num_cols[2]:
            st.empty()
    else:
        num_cols = st.columns(1)
        with num_cols[0]:
            st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:600;'>AC/IRRA: {inputs['AC/IRRA']:.4f}</div>",unsafe_allow_html=True)

        # -- Second row: ALE plots
    if not show_ac_slider:
        plot_cols = st.columns(3)
        ratio_features = ["DC/IRRA", "AC/IRRA"]
    else:
        plot_cols = st.columns(3)
        ratio_features = ["AC/IRRA"]
            
    for idx, feat in enumerate(ratio_features):
        col = plot_cols[idx]
        val = inputs[feat]
        with col:
            # Only show plot if ALE exists
            if feat not in ale_df["feature"].unique():
                st.info("No ALE PLOT")
                continue
                # Cache key
            fig = draw_ale_mini_plot(ale_df, feat, val)
            st.plotly_chart(fig, use_container_width=True)


    X_input = pd.DataFrame([inputs], columns=features)

    if st.button("Predict"):
        score = float(svm.decision_function(X_input))
        label = 1 if score >= thr else 0
        st.markdown(f"### Prediction: **{'Suboptimal (1)' if label else 'Optimal (0)'}**")
        st.markdown(f"Score: `{score:.4f}`, Threshold: `{thr:.4f}`")
        st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=score,gauge={"axis": {"range": [-3, 3]}, "threshold": {"value": thr}},title={"text": "SVM Decision Score"})),use_container_width=True,)

    # -------------------------
    # Tabs
    # -------------------------
    st.markdown("---")
    st.header("🧾 PERFORMANCE METRICS")
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "Metrics", "SVM Score Distribution", "Feature Importance (Drop-Column)"])

    with tab1:
        st.header("Confusion Matrix")
        st.plotly_chart(make_cm_fig(m["cm_svm"]), use_container_width=True)

    with tab2:
        st.header("Performance Metrics")
        row = m["metrics"].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{row['precision']:.3f}")
        c2.metric("Recall", f"{row['recall']:.3f}")
        c3.metric("F1 Score", f"{row['f1']:.3f}")
        st.markdown("### Full Metrics Table")
        st.dataframe(m["metrics"])

    with tab3:
        st.header("SVM Decision Function Score Distribution")
        X_te_df = pd.DataFrame(m["X_te"], columns=features)
        scores = svm.decision_function(X_te_df)
        st.plotly_chart(make_hist(scores, m["y_te"], thr, axis_title_size=20, font_size=18), use_container_width=True)

    with tab4:
        st.header("Drop-Column Importance (SVM)")
        drop_df = m["drop_importance"].sort_values("importance", ascending=False)
        fig = px.bar(drop_df, x="feature", y="importance",title="Change in F1 Score When Feature is Removed",text_auto=".3f", color="importance", color_continuous_scale="Teal", )
        fig.update_layout(xaxis_tickangle=45, margin=dict(l=40, r=40, t=60, b=120), font=dict(size=16))
        mn, mx = drop_df["importance"].min(), drop_df["importance"].max()
        fig.update_yaxes(range=[mn - abs(mn) * 2, mx + abs(mx) * 0.2])
        st.plotly_chart(apply_plot_style(fig), use_container_width=True)
    # -------------------------
    # Threshold Explorer
    # -------------------------
    st.markdown("---")
    st.header("🎚 THRESHOLD EXPLORER ")
    st.markdown(" ")

    X_te_df = pd.DataFrame(m["X_te"], columns=features)
    y_true = m["y_te"]
    scores = svm.decision_function(X_te_df)

    st.write("#### Adjust Threshold :")
    thr_user = st.slider("", float(np.min(scores)), float(np.max(scores)), float(thr), step=0.01, label_visibility="collapsed")
    # thr_user = st.slider("Adjust Threshold", float(np.min(scores)), float(np.max(scores)), float(thr), step=0.01)
    pred_user = (scores >= thr_user).astype(int)
    st.write(" ") 
    
    p = precision_score(y_true, pred_user, zero_division=0)
    r = recall_score(y_true, pred_user, zero_division=0)
    f1 = f1_score(y_true, pred_user, zero_division=0)
    cm_u = confusion_matrix(y_true, pred_user)
    st.write(" ")
    c1, c2, c3 = st.columns(3)
    
    def compact_metric(label, value):
        st.markdown(f"""<div style="text-align:center; line-height:2;"><div style="font-size:30px; font-weight:600; margin-bottom:4px;"> {label}</div>
                <div style="font-size:32px; font-weight:700;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True)
    with c1:
        compact_metric("Precision :", f"{p:.3f}")

    with c2:
        compact_metric("Recall :", f"{r:.3f}")

    with c3:
        compact_metric("F1 Score :", f"{f1:.3f}")

    #c1.metric("Precision", f"{p:.3f}")
    #c2.metric("Recall", f"{r:.3f}")
    #c3.metric("F1 Score", f"{f1:.3f}")
    st.write(" ")    
    st.subheader("▫️ Score Distribution")
    st.plotly_chart(make_hist(scores, y_true, thr_user), use_container_width=True, key="threshold_hist")

    st.subheader("▫️ Confusion Matrix")
    st.plotly_chart(apply_plot_style(make_cm_fig(cm_u)), use_container_width=True, key="threshold_cm")


# -------------------------
# DASHBOARD 3
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "lresults")

RESULT_FILES = {
    "Plant 1": {
        "AC": os.path.join(BASE_DIR, "results_p1_ac.csv"),
        "DC": os.path.join(BASE_DIR, "results_p1_dc.csv"),
    },
    "Plant 2": {
        "AC": os.path.join(BASE_DIR, "results_p2_ac.csv"),
        "DC": os.path.join(BASE_DIR, "results_p2_dc.csv"),
    },
}

# 新增：时序数据文件（刚刚导出的 *_ts.csv）
TS_FILES = {
    "Plant 1": {
        "AC": os.path.join(BASE_DIR, "p1_ac_ts.csv"),
        "DC": os.path.join(BASE_DIR, "p1_dc_ts.csv"),
    },
    "Plant 2": {
        "AC": os.path.join(BASE_DIR, "p2_ac_ts.csv"),
        "DC": os.path.join(BASE_DIR, "p2_dc_ts.csv"),
    },
}

PLANTS       = ["Plant 1", "Plant 2"]
TARGET_TYPES = ["AC", "DC"]

MODEL_NAME_MAP = {
    "LSTM_MAE":  "LSTM",
    "Pers_MAE":  "Persistence",
    "MA_MAE":    "Moving Avg",
    "LSTM_RMSE": "LSTM",
    "Pers_RMSE": "Persistence",
    "MA_RMSE":   "Moving Avg",
}

#%% ================== 读取 Task 6 结果 ==================

def load_results_df(plant: str, target_type: str) -> pd.DataFrame:
    path = RESULT_FILES[plant][target_type]
    if not os.path.exists(path):
        raise FileNotFoundError(f"结果文件没找到：{path}")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns and "inverter" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "inverter"})
    return df

# 新增：读取时序数据
def load_timeseries_df(plant: str, target_type: str) -> pd.DataFrame:
    """
    读取 export_timeseries_for_plant 保存的 *_ts.csv，
    其中包含列：
      AC_TRUE / DC_TRUE
      AC_LSTM / DC_LSTM
      AC_PERSIST / DC_PERSIST
      AC_MA / DC_MA
      IRRADIATION
      date
    """
    path = TS_FILES[plant][target_type]
    if not os.path.exists(path):
        raise FileNotFoundError(f"时序文件没找到：{path}")
    df = pd.read_csv(path, parse_dates=["Time"], index_col="Time")
    return df

#%% ================== Streamlit App ==================


def dashboard_3():
    st.title("LSTM Model")

    # ----------- Sidebar：选择 plant & target -----------

    st.sidebar.header("🔧 Controls")

    plant = st.sidebar.selectbox("Select plant:", PLANTS)
    target_type = st.sidebar.radio("Select target type:", TARGET_TYPES, horizontal=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Metrics are computed from your Task 6 LSTM + baseline results.")

    # ----------- 主区域：加载结果并展示 -----------

    st.header(f"📊 Offline performance — {plant}, {target_type}")

    # 1. 读结果表
    results_df = load_results_df(plant, target_type)

    st.subheader("Raw metrics per inverter")
    st.dataframe(results_df, use_container_width=True)

    # 2. 计算所有 inverter 的平均 MAE / RMSE
    metric_cols_mae  = ["LSTM_MAE", "Pers_MAE", "MA_MAE"]
    metric_cols_rmse = ["LSTM_RMSE", "Pers_RMSE", "MA_RMSE"]

    avg_mae  = results_df[metric_cols_mae].mean()
    avg_rmse = results_df[metric_cols_rmse].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Average MAE across inverters")
        for col in metric_cols_mae:
            st.metric(MODEL_NAME_MAP[col], f"{avg_mae[col]:.2f}")
    with col2:
        st.markdown("### Average RMSE across inverters")
        for col in metric_cols_rmse:
            st.metric(MODEL_NAME_MAP[col.replace("MAE", "RMSE")], f"{avg_rmse[col]:.2f}")

    # 3. 按 inverter 画 MAE 条形图
    st.subheader("🔍 MAE by inverter and model")

    df_mae_long = results_df.melt(
        id_vars=["inverter"],
        value_vars=metric_cols_mae,
        var_name="Model_raw",
        value_name="MAE",
    )
    df_mae_long["Model"] = df_mae_long["Model_raw"].map(MODEL_NAME_MAP)

    fig_mae = px.bar(
        df_mae_long,
        x="inverter",
        y="MAE",
        color="Model",
        barmode="group",
        title=f"MAE comparison for {plant} — {target_type}",
        text_auto=".1f",
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    # 4. 按 inverter 画 RMSE 条形图
    st.subheader("🔍 RMSE by inverter and model")

    df_rmse_long = results_df.melt(
        id_vars=["inverter"],
        value_vars=metric_cols_rmse,
        var_name="Model_raw",
        value_name="RMSE",
    )
    df_rmse_long["Model"] = df_rmse_long["Model_raw"].map(MODEL_NAME_MAP)

    fig_rmse = px.bar(
        df_rmse_long,
        x="inverter",
        y="RMSE",
        color="Model",
        barmode="group",
        title=f"RMSE comparison for {plant} — {target_type}",
        text_auto=".1f",
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    # ===================== 时间序列互动图 =====================

    st.subheader("📈 Time-series for best inverter (interactive)")

    show_ts = st.checkbox("Show interactive time-series plot", value=False)

    if show_ts:
        try:
            ts_df = load_timeseries_df(plant, target_type)
        except FileNotFoundError as e:
            st.warning(str(e))
        else:
            target_prefix = target_type.upper()  # "AC" or "DC"
            col_true    = f"{target_prefix}_TRUE"
            col_lstm    = f"{target_prefix}_LSTM"
            col_persist = f"{target_prefix}_PERSIST"
            col_ma      = f"{target_prefix}_MA"

            # ---- 控制哪些曲线显示：这里就是你要的“按钮” ----
            st.markdown("**Select which lines to show:**")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                show_true = st.checkbox("Actual", value=True, key="ts_true")
            with c2:
                show_lstm = st.checkbox("LSTM", value=True, key="ts_lstm")
            with c3:
                show_pers = st.checkbox("Persistence", value=False, key="ts_pers")
            with c4:
                show_ma   = st.checkbox("Moving Avg", value=False, key="ts_ma")
            with c5:
                show_irr  = st.checkbox("Irradiation", value=False, key="ts_irr")

            # ---- 用 Plotly 画线：按按钮决定加哪条 trace ----
            fig_ts = px.line()
            fig_ts.update_layout(
                title=f"{plant} – {target_type} power (best inverter)",
                xaxis_title="Time",
                yaxis_title=f"{target_prefix} Power",
            )

            if show_true and col_true in ts_df.columns:
                fig_ts.add_scatter(
                    x=ts_df.index, y=ts_df[col_true],
                    mode="lines",
                    name="Actual",
                )
            if show_lstm and col_lstm in ts_df.columns:
                fig_ts.add_scatter(
                    x=ts_df.index, y=ts_df[col_lstm],
                    mode="lines",
                    name="LSTM",
                )
            if show_pers and col_persist in ts_df.columns:
                fig_ts.add_scatter(
                    x=ts_df.index, y=ts_df[col_persist],
                    mode="lines",
                    name="Persistence",
                )
            if show_ma and col_ma in ts_df.columns:
                fig_ts.add_scatter(
                    x=ts_df.index, y=ts_df[col_ma],
                    mode="lines",
                    name="Moving Avg",
                )
            # Irradiation 单位不一样，可以放到 secondary y（简单起见，这里直接画在一起）
            if show_irr and "IRRADIATION" in ts_df.columns:
                fig_ts.add_scatter(
                    x=ts_df.index, y=ts_df["IRRADIATION"],
                    mode="lines",
                    name="Irradiation",
                )

            st.plotly_chart(fig_ts, use_container_width=True)

            st.caption(
                "Each checkbox above corresponds to one time-series line. "
                "Turning them on/off lets the user inspect how LSTM compares with "
                "persistence and moving-average baselines, and how irradiation drives power."
            )

            # ===================== 代表性晴天 / 多云天 =====================
            st.subheader("☀️ Representative sunny & cloudy days")

            # 阈值：用实际功率 > threshold 定义“白天”
            power_threshold = st.number_input(
                "Daytime power threshold to define 'day' [kW]",
                min_value=0.0,
                max_value=500.0,
                value=50.0,
                step=10.0,
            )

            # daytime 只看实际功率
            daytime_mask = ts_df[col_true] > power_threshold
            daytime_df = ts_df.loc[daytime_mask]

            if daytime_df.empty:
                st.warning(
                    "No daytime data above the given threshold. "
                    "Try lowering the threshold."
                )
            else:
                # 用实际功率的标准差来区分稳定 / 多云（不再用 irradiation）
                day_std = daytime_df.groupby("date")[col_true].std().dropna()

                if len(day_std) == 0:
                    st.warning(
                        "Cannot distinguish sunny / cloudy days "
                        "(no valid std per day)."
                    )
                else:
                    days_to_plot = []
                    if len(day_std) >= 2:
                        stable_day = day_std.idxmin()
                        cloudy_day = day_std.idxmax()
                        if stable_day == cloudy_day:
                            days_to_plot = [(stable_day, "Representative Day")]
                        else:
                            days_to_plot = [
                                (stable_day, "Sunny & Stable Day"),
                                (cloudy_day, "Cloudy & Variable Day"),
                            ]
                    else:
                        only_day = day_std.index[0]
                        days_to_plot = [(only_day, "Single Representative Day")]

                    # 逐个代表性日期画图
                    for day, title_txt in days_to_plot:
                        sub = ts_df[ts_df["date"] == day]
                        if sub.empty:
                            continue

                        day_str = str(day)
                        st.markdown(f"#### {title_txt} — {day_str}")

                        # 每个 day 自己的按钮组，key 里带 day_str 保证唯一
                        c1d, c2d, c3d, c4d = st.columns(4)
                        with c1d:
                            show_true_d = st.checkbox(
                                "Actual", value=True, key=f"{title_txt}_{day_str}_true"
                            )
                        with c2d:
                            show_lstm_d = st.checkbox(
                                "LSTM", value=True, key=f"{title_txt}_{day_str}_lstm"
                            )
                        with c3d:
                            show_pers_d = st.checkbox(
                                "Persistence", value=False, key=f"{title_txt}_{day_str}_pers"
                            )
                        with c4d:
                            show_ma_d = st.checkbox(
                                "Moving Avg", value=False, key=f"{title_txt}_{day_str}_ma"
                            )

                        fig_day = px.line()
                        fig_day.update_layout(
                            title=f"{title_txt} — {plant} — {target_type} — {day_str}",
                            xaxis_title="Time",
                            yaxis_title=f"{target_prefix} Power",
                        )

                        if show_true_d and col_true in sub.columns:
                            fig_day.add_scatter(
                                x=sub.index,
                                y=sub[col_true],
                                mode="lines",
                                name="Actual",
                            )
                        if show_lstm_d and col_lstm in sub.columns:
                            fig_day.add_scatter(
                                x=sub.index,
                                y=sub[col_lstm],
                                mode="lines",
                                name="LSTM",
                            )
                        if show_pers_d and col_persist in sub.columns:
                            fig_day.add_scatter(
                                x=sub.index,
                                y=sub[col_persist],
                                mode="lines",
                                name="Persistence",
                            )
                        if show_ma_d and col_ma in sub.columns:
                            fig_day.add_scatter(
                                x=sub.index,
                                y=sub[col_ma],
                                mode="lines",
                                name="Moving Avg",
                            )

                        st.plotly_chart(fig_day, use_container_width=True)

                        st.caption(
                            "Standard deviation of actual power over daytime defines "
                            "which days are 'sunny & stable' vs 'cloudy & variable'. "
                            "This reproduces the idea from visualize_lstm_results()."
                        )


page = st.sidebar.selectbox("Choose a Model",['Regression Model', "Classification Model", "LSTM Model"])

if page == 'Regression Model':
    dashboard_1()
    
elif page == "Classification Model":
    dashboard_2()

elif page == "LSTM Model":
    dashboard_3()
