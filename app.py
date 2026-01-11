from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np

import plotly.graph_objs as go

from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import make_regression

app = Flask(__name__)
# PART 1: For the Bar, not yet!
# -----------------------------



# PART 2: Upload and Models
# --------------------------
# In-memory dataset (for your local demo)
current_df: pd.DataFrame | None = None


def _read_table_from_upload(file_storage) -> pd.DataFrame:
    filename = (file_storage.filename or "").lower()
    if filename.endswith(".xlsx"):
        # requires openpyxl installed
        df = pd.read_excel(file_storage)
    else:
        df = pd.read_csv(file_storage)
    return df


def _preview_html(df: pd.DataFrame, n: int = 10) -> str:
    return df.head(n).to_html(classes="data-table", index=False, border=0)


def _numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload_data")
def upload_data():
    global current_df
    file = request.files.get("datafile")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        df = _read_table_from_upload(file)
        if df.empty:
            return jsonify({"error": "File loaded but it is empty."}), 400
        current_df = df
        return jsonify({
            "ok": True,
            "preview_html": _preview_html(df, 10),
            "columns": df.columns.tolist()
        })
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 400


@app.post("/load_example")
def load_example():
    """
    Example dataset like you requested: 100 rows, 4 features.
    """
    global current_df
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "feature1": rng.normal(0, 1, 100),
        "feature2": rng.normal(2, 1.5, 100),
        "feature3": rng.normal(-1, 0.75, 100),
        "feature4": rng.normal(5, 2.0, 100),
    })
    # synthetic target (continuous)
    df["target"] = 2.2*df["feature1"] - 0.8*df["feature2"] + 0.5*df["feature3"] + rng.normal(0, 0.5, 100)

    current_df = df
    return jsonify({
        "ok": True,
        "preview_html": _preview_html(df, 10),
        "columns": df.columns.tolist(),
        "shape": [df.shape[0], df.shape[1]]
    })


@app.post("/model_metrics")
def model_metrics():
    """
    Returns metrics for one model type (lr/svm/nn/knn/cluster).
    """
    global current_df
    if current_df is None:
        return jsonify({"error": "No dataset loaded. Upload a file or use the example dataset."}), 400

    data = request.get_json(force=True)
    model_type = (data.get("model_type") or "").strip().lower()
    features = data.get("features") or []
    target = (data.get("target") or "").strip()
    n_clusters = int(data.get("n_clusters") or 3)
    n_neighbors = int(data.get("n_neighbors") or 5)

    if not features:
        return jsonify({"error": "Enter feature columns first."}), 400

    for c in features:
        if c not in current_df.columns:
            return jsonify({"error": f"Feature column not found: {c}"}), 400

    # clustering doesn’t require target
    if model_type == "cluster":
        X = current_df[features].copy()
        X = pd.get_dummies(X, drop_first=True)
        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        if len(X) < 10:
            return jsonify({"error": "Not enough clean numeric rows for clustering."}), 400

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        counts = pd.Series(labels).value_counts().sort_index().to_dict()
        return jsonify({
            "ok": True,
            "model_type": "Clusters",
            "rmse": None,
            "mae": None,
            "r2": None,
            "notes": f"Inertia: {km.inertia_:.3f}, cluster counts: {counts}"
        })

    # regression models require a target
    if not target:
        return jsonify({"error": "Enter a target column for LR/SVM/NN/KNN."}), 400
    if target not in current_df.columns:
        return jsonify({"error": f"Target column not found: {target}"}), 400

    df = current_df[features + [target]].copy()
    df = df.dropna()

    # numeric features for regression
    X = _numeric_df(df, features).dropna()
    y = pd.to_numeric(df.loc[X.index, target], errors="coerce").dropna()

    # align
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]

    if len(X) < 10:
        return jsonify({"error": "Not enough numeric rows after cleaning. Try different columns."}), 400

    # simple split
    n = len(X)
    split = int(0.8 * n)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type == "lr":
        model = LinearRegression()
        label = "Linear Regression"
    elif model_type == "svm":
        model = SVR()
        label = "SVM"
    elif model_type == "nn":
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)
        label = "Neural Network"
    elif model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        label = f"KNN (k={n_neighbors})"
    else:
        return jsonify({"error": f"Unknown model_type: {model_type}"}), 400

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(model.score(X_test, y_test))

    return jsonify({
        "ok": True,
        "model_type": label,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "notes": f"Train rows: {len(X_train)}, Test rows: {len(X_test)}"
    })


@app.post("/model_fit_2d")
def model_fit_2d():
    """
    Overlay multiple models on the SAME 2D plot.
    x_feature = first feature in features[]
    y = target (or second axis for cluster visualization)
    """
    global current_df
    if current_df is None:
        return jsonify({"error": "No dataset loaded. Upload a file or use the example dataset."}), 400

    payload = request.get_json(force=True)
    models = payload.get("models") or []
    features = payload.get("features") or []
    target = (payload.get("target") or "").strip()
    n_clusters = int(payload.get("n_clusters") or 3)
    n_neighbors = int(payload.get("n_neighbors") or 5)

    if len(features) < 1:
        return jsonify({"error": "Need at least 1 feature for 2D plot."}), 400
    x_feature = features[0]
    if x_feature not in current_df.columns:
        return jsonify({"error": f"Feature not found: {x_feature}"}), 400

    # Build x
    x = pd.to_numeric(current_df[x_feature], errors="coerce")
    if x.dropna().shape[0] < 10:
        return jsonify({"error": "Not enough numeric data in the first feature for plotting."}), 400

    fig = go.Figure()
    fig.update_layout(
        title="2D Model Overlay",
        xaxis_title=x_feature,
        yaxis_title=target if target else "Value",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        height=520,
    )

    # If target exists, plot data points (x vs y)
    if target and target in current_df.columns:
        y = pd.to_numeric(current_df[target], errors="coerce")
        df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
        df_xy = df_xy.sort_values("x")

        fig.add_trace(go.Scatter(
            x=df_xy["x"],
            y=df_xy["y"],
            mode="markers",
            name="Data",
        ))

        # fit lines need sorted x
        x_sorted = df_xy["x"].to_numpy()
        X_line = x_sorted.reshape(-1, 1)

        for m in models:
            m = (m or "").lower()
            if m == "cluster":
                # clustering shown as colored markers (no line)
                XY = df_xy[["x", "y"]].to_numpy()
                km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                labels = km.fit_predict(XY)
                fig.add_trace(go.Scatter(
                    x=df_xy["x"],
                    y=df_xy["y"],
                    mode="markers",
                    name=f"Clusters (k={n_clusters})",
                    marker=dict(color=labels),
                ))
                continue

            if m == "lr":
                model = LinearRegression()
                name = "LR Fit"
            elif m == "svm":
                model = SVR()
                name = "SVM Fit"
            elif m == "nn":
                model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)
                name = "NN Fit"
            elif m == "knn":
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
                name = f"KNN Fit (k={n_neighbors})"
            else:
                continue

            # Train on full 2D relationship for the overlay
            X_train = df_xy["x"].to_numpy().reshape(-1, 1)
            y_train = df_xy["y"].to_numpy()
            model.fit(X_train, y_train)
            y_line = model.predict(X_line)

            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=y_line,
                mode="lines",
                name=name,
            ))

        return jsonify({"ok": True, "figure": fig.to_dict()})

    # If no target provided: just show x distribution
    df_x = pd.DataFrame({"x": x}).dropna().sort_values("x")
    fig.add_trace(go.Histogram(x=df_x["x"], nbinsx=18, name="Feature distribution"))
    return jsonify({"ok": True, "figure": fig.to_dict()})


@app.post("/plot3d")
def plot3d():
    """
    Only called when user enters 3 feature columns.
    Colors by target if provided; otherwise plain points.
    """
    global current_df
    if current_df is None:
        return jsonify({"error": "No dataset loaded."}), 400

    payload = request.get_json(force=True)
    features = payload.get("features") or []
    target = (payload.get("target") or "").strip()

    if len(features) != 3:
        return jsonify({"error": "3D plot requires exactly 3 features."}), 400

    for c in features:
        if c not in current_df.columns:
            return jsonify({"error": f"Feature not found: {c}"}), 400

    df = current_df[features].copy()
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    if len(df) < 10:
        return jsonify({"error": "Not enough numeric rows for 3D plot."}), 400

    marker = dict(size=4, opacity=0.85)

    if target and target in current_df.columns:
        t = current_df.loc[df.index, target]
        # allow categorical or numeric
        if t.dtype == "object":
            codes, _ = pd.factorize(t.astype(str))
            marker["color"] = codes
        else:
            marker["color"] = pd.to_numeric(t, errors="coerce").fillna(0).to_numpy()

    fig = go.Figure(data=[
        go.Scatter3d(
            x=df[features[0]],
            y=df[features[1]],
            z=df[features[2]],
            mode="markers",
            marker=marker
        )
    ])
    fig.update_layout(
        title="3D Scatter",
        scene=dict(
            xaxis_title=features[0],
            yaxis_title=features[1],
            zaxis_title=features[2],
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        height=560
    )

    return jsonify({"ok": True, "figure": fig.to_dict()})

# ----------------------------------
# PART 3: HISTOGRAM IS JUST A GRAPH
# ----------------------------------
@app.route("/histogram", methods=["POST"])
def histogram():
    global current_df

    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400

    data = request.get_json()
    column = data.get("column")

    if column not in current_df.columns:
        return jsonify({"error": f"Column '{column}' not found"}), 400

    values = current_df[column].dropna().values

    if len(values) < 5:
        return jsonify({"error": "Not enough data"}), 400

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=12,
        marker=dict(color="rgba(120,140,255,0.75)")
    ))

    fig.update_layout(
        title=f"Histogram – {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
        template="plotly_dark",
        bargap=0.15
    )

    return jsonify({
        "figure": fig.to_dict(),
        "stats": {
            "mean": mean,
            "std": std,
            "n": len(values)
        }
    })


# PART 4: GAUGE R&R CALCULATION OVERVIEW 
# ---------------------------------------
@app.route("/gauge_rr", methods=["POST"])
def gauge_rr():
    data = request.get_json()
    features = data.get("features", [])

    df = current_df[features].dropna()

    # Means & ranges per "operator"
    means = df.mean()
    ranges = df.max() - df.min()

    # Repeatability (EV): average range
    EV = ranges.mean()

    # Reproducibility (AV): std of feature means
    AV = means.std(ddof=1)

    # Total Gauge R&R
    GRR = np.sqrt(EV**2 + AV**2)

    return jsonify({
        "means": means.to_dict(),
        "ranges": ranges.to_dict(),
        "repeatability": EV,
        "reproducibility": AV,
        "total_grr": GRR,
        "note": (
            "With the Total Gauge R&R, the observation of variation "
            "is found in the measurement of the system itself. So, a high GRR would "
            "mean unreliable decisions."
        )
    })


# PART 5: Control Chart Limits
# -----------------------------
@app.route("/control_limits", methods=["POST"])
def control_limits():
    global current_df

    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400

    df = current_df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        return jsonify({"error": "Need at least 2 numeric columns"}), 400

    # Treat columns as subgroups
    means = df.mean()
    ranges = df.max() - df.min()

    Xbarbar = means.mean()
    Rbar = ranges.mean()

    n = df.shape[0]
    A2 = 1.88 if n == 2 else 1.023 if n == 3 else 0.577  # safe demo fallback

    UCL = Xbarbar + A2 * Rbar
    LCL = Xbarbar - A2 * Rbar

    return jsonify({
        "means": means.round(4).to_dict(),
        "ranges": ranges.round(4).to_dict(),
        "Xbarbar": round(Xbarbar, 4),
        "Rbar": round(Rbar, 4),
        "UCL": round(UCL, 4),
        "CL": round(Xbarbar, 4),
        "LCL": round(LCL, 4)
    })


# PART 6: Process Capability (Cp & Cpk)
# --------------------------------------
@app.route("/cpk", methods=["POST"])
def cpk():
    global current_df

    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400

    data = request.get_json()
    lsl = float(data.get("lsl"))
    usl = float(data.get("usl"))

    df = current_df.select_dtypes(include=[np.number])
    values = df.values.flatten()

    mean = values.mean()
    std = values.std(ddof=1)

    cp = (usl - lsl) / (6 * std)
    cpk = min((usl - mean), (mean - lsl)) / (3 * std)

    # ---- Industry judgment ----
    if cpk >= 1.33:
        judgment = "Pass"
        judgment_color = "green"
    elif cpk >= 1.0:
        judgment = "Marginal"
        judgment_color = "orange"
    else:
        judgment = "Fail"
        judgment_color = "red"

    return jsonify({
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cp": round(cp, 4),
        "cpk": round(cpk, 4),
        "judgment": judgment,
        "judgment_color": judgment_color,
        "values": values.tolist()  # for histogram
    })

# PART 7: Interactive X-R Control Charts 
# ---------------------------------------
@app.route("/xr_chart", methods=["POST"])
def xr_chart():
    global current_df

    df = current_df.select_dtypes(include=[np.number])
    data = df.values

    means = data.mean(axis=1)
    ranges = data.max(axis=1) - data.min(axis=1)

    x = list(range(1, len(means) + 1))

    return jsonify({
        "subgroup": x,
        "means": means.tolist(),
        "ranges": ranges.tolist()
    })



@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run()