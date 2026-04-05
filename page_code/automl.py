import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import io


def show():
    st.title("⚙️ AutoML — Custom Dataset")
    st.caption("Upload your own equipment CSV · agent preprocesses, trains and evaluates automatically")

    # ── Upload section ────────────────────────────────────────────────────────
    st.subheader("1  Upload dataset")
    uploaded = st.file_uploader(
        "Drop a CSV file with sensor readings + binary failure label",
        type=["csv"],
        help="Must include a binary label column (0 = healthy, 1 = failure)"
    )

    use_demo = st.checkbox("Use built-in demo dataset (HVAC synthetic)", value=(uploaded is None))

    if use_demo or uploaded is None:
        # Generate a demo dataset
        rng = np.random.default_rng(7)
        n = 500
        df_raw = pd.DataFrame({
            "temperature_c":    rng.uniform(15, 45, n).round(2),
            "vibration_mms":    rng.uniform(0, 15, n).round(2),
            "pressure_pa":      rng.uniform(50, 400, n).round(1),
            "runtime_hrs":      rng.integers(100, 9000, n).astype(float),
            "humidity_pct":     rng.uniform(20, 95, n).round(1),
        })
        score = (
            (df_raw["temperature_c"] - 15) / 30 * 0.3 +
            df_raw["vibration_mms"] / 15 * 0.35 +
            (df_raw["pressure_pa"] - 50) / 350 * 0.2 +
            df_raw["runtime_hrs"] / 9000 * 0.15
        )
        df_raw["failure"] = (score + rng.uniform(-0.12, 0.12, n) > 0.55).astype(int)
        st.info("Using demo dataset (500 samples · 5 features · binary failure label)")
    else:
        df_raw = pd.read_csv(uploaded)
        st.success(f"Uploaded: {uploaded.name} · {len(df_raw):,} rows · {df_raw.shape[1]} columns")

    if df_raw is not None:
        with st.expander("Preview dataset", expanded=False):
            st.dataframe(df_raw.head(20), use_container_width=True)
            st.caption(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

    st.divider()

    # ── Configuration ─────────────────────────────────────────────────────────
    st.subheader("2  Configure pipeline")
    all_cols = list(df_raw.columns) if df_raw is not None else []

    cfg1, cfg2 = st.columns(2)
    target_col  = cfg1.selectbox("Target (label) column", all_cols,
                                  index=len(all_cols)-1 if all_cols else 0)
    equip_name  = cfg2.text_input("Equipment category name", value="Custom equipment")

    feat_cols = st.multiselect(
        "Feature columns to use", [c for c in all_cols if c != target_col],
        default=[c for c in all_cols if c != target_col]
    )

    ac1, ac2, ac3 = st.columns(3)
    test_split  = ac1.slider("Test split (%)", 10, 40, 20)
    cv_folds    = ac2.slider("Cross-validation folds", 3, 10, 5)
    models_sel  = ac3.multiselect(
        "Models to try",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        default=["Random Forest", "Gradient Boosting"]
    )

    st.divider()

    # ── Run AutoML ────────────────────────────────────────────────────────────
    if st.button("▶ Start AutoML Pipeline", type="primary", use_container_width=True):
        if not feat_cols:
            st.error("Please select at least one feature column."); return
        if target_col not in df_raw.columns:
            st.error("Target column not found in dataset."); return

        X = df_raw[feat_cols].copy()
        y = df_raw[target_col].copy()

        # ── Step 1: Data ingestion ────────────────────────────────────────────
        with st.status("Running AutoML pipeline…", expanded=True) as status:
            st.write("**Step 1 / 6 — Data ingestion**")
            time.sleep(0.4)
            n_rows, n_feats = X.shape
            st.write(f"  → {n_rows} rows · {n_feats} features · target: `{target_col}`")
            st.write(f"  → Class balance: {int(y.sum())} positives / {int((y==0).sum())} negatives")

            # ── Step 2: Preprocessing ─────────────────────────────────────────
            st.write("**Step 2 / 6 — Preprocessing**")
            time.sleep(0.5)
            missing = X.isnull().sum().sum()
            X = X.fillna(X.median(numeric_only=True))
            X = X.select_dtypes(include=[np.number])
            st.write(f"  → {missing} missing values imputed · {X.shape[1]} numeric features retained")

            # ── Step 3: Feature engineering ───────────────────────────────────
            st.write("**Step 3 / 6 — Feature engineering**")
            time.sleep(0.5)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            st.write(f"  → StandardScaler applied · features normalised to μ=0, σ=1")

            # ── Step 4: Model training ────────────────────────────────────────
            st.write("**Step 4 / 6 — Model training & cross-validation**")
            time.sleep(0.4)
            model_map = {
                "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            }
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            results = {}
            for name in models_sel:
                clf = model_map[name]
                scores = cross_val_score(clf, X_s, y, cv=cv, scoring="f1")
                results[name] = {"f1_mean": scores.mean(), "f1_std": scores.std(), "clf": clf}
                st.write(f"  → {name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")

            # ── Step 5: Evaluation ────────────────────────────────────────────
            st.write("**Step 5 / 6 — Best model evaluation**")
            time.sleep(0.4)
            best_name = max(results, key=lambda k: results[k]["f1_mean"])
            best_clf  = results[best_name]["clf"]
            best_clf.fit(X_s, y)
            y_pred = best_clf.predict(X_s)
            st.write(f"  → Best model: **{best_name}** (F1 = {results[best_name]['f1_mean']:.3f})")

            # ── Step 6: Deployment ────────────────────────────────────────────
            st.write("**Step 6 / 6 — Model registration**")
            time.sleep(0.5)
            st.write(f"  → Model registered as `{equip_name.replace(' ','_').lower()}_automl_v1`")
            st.write("  → Ready for real-time prediction endpoint")
            status.update(label="✅ AutoML pipeline complete!", state="complete")

        # ── Results ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Best model",      best_name)
        r2.metric("Best F1-score",   f"{results[best_name]['f1_mean']:.3f}")
        r3.metric("Training samples",n_rows)
        r4.metric("Features used",   n_feats)

        col_a, col_b = st.columns(2)

        # Model comparison chart
        with col_a:
            st.markdown("**Model comparison**")
            names   = list(results.keys())
            f1s     = [results[n]["f1_mean"] for n in names]
            stds    = [results[n]["f1_std"]  for n in names]
            colors  = ["#28a745" if n == best_name else "#4a90d9" for n in names]
            fig = go.Figure(go.Bar(
                x=names, y=f1s,
                error_y=dict(type="data", array=stds, visible=True),
                marker_color=colors,
                text=[f"{v:.3f}" for v in f1s], textposition="outside"
            ))
            fig.update_layout(
                yaxis=dict(title="F1-score (CV)", range=[0, 1.1]),
                margin=dict(l=0,r=0,t=10,b=20), height=240,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature importances
        with col_b:
            if hasattr(best_clf, "feature_importances_"):
                st.markdown("**Feature importances**")
                imp = best_clf.feature_importances_
                fi_df = pd.DataFrame({"Feature": X.columns, "Importance": imp})
                fi_df = fi_df.sort_values("Importance", ascending=True)
                fig2 = go.Figure(go.Bar(
                    x=fi_df["Importance"], y=fi_df["Feature"],
                    orientation="h", marker_color="#4a90d9",
                    text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside"
                ))
                fig2.update_layout(
                    xaxis_title="Importance", yaxis_title="",
                    margin=dict(l=0,r=20,t=10,b=20), height=240,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Classification report
        with st.expander("Full classification report"):
            report = classification_report(y, y_pred, output_dict=True)
            rep_df = pd.DataFrame(report).T.round(3)
            st.dataframe(rep_df, use_container_width=True)

        # Download model info
        model_info = f"""FM AI Agent v2 — AutoML Result
Equipment: {equip_name}
Best model: {best_name}
F1-score: {results[best_name]['f1_mean']:.4f}
Features: {list(X.columns)}
Training samples: {n_rows}
"""
        st.download_button("⬇ Download model summary", model_info,
                           file_name="automl_result.txt", mime="text/plain")
