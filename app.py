import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Interactive Regression Explorer", layout="wide")

st.title("📈 Interactive Regression Explorer (MLE & MAP – OLS, Ridge, Lasso, Elastic Net)")

st.markdown("""
This app lets you **upload a dataset**, choose **regression methods**, and see:

- OLS / Ridge / Lasso / Elastic Net
- MLE and MAP interpretation
- Coefficient tables (numeric solution)
- Correlation matrix
- 2D and 3D visualizations
- Gaussian residual plots
""")


# ===================== 1. DATA LOADING =====================

st.sidebar.header("1. Data options")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Sample: Diabetes dataset", "Upload CSV"]
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or use the sample dataset.")
        st.stop()
else:
    # Sample: sklearn diabetes dataset (regression)
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame

st.subheader("Dataset preview")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns (1 target + 1 feature).")
    st.stop()


# ===================== 2. TARGET & FEATURES =====================

st.sidebar.header("2. Features & Target")

target_col = st.sidebar.selectbox("Select target (y)", numeric_cols, index=len(numeric_cols)-1)
feature_cols = st.sidebar.multiselect(
    "Select feature columns (X)", [c for c in numeric_cols if c != target_col],
    default=[c for c in numeric_cols if c != target_col][:2]
)

if not feature_cols:
    st.error("Please select at least one feature column.")
    st.stop()

X = df[feature_cols].values
y = df[target_col].values

test_size = st.sidebar.slider("Test size (for metrics)", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)


# ===================== 3. MODEL CHOICES =====================

st.sidebar.header("3. Regression methods")

methods = st.sidebar.multiselect(
    "Select methods",
    ["OLS", "Ridge", "Lasso", "Elastic Net"],
    default=["OLS", "Ridge", "Lasso", "Elastic Net"]
)

ridge_alpha = st.sidebar.number_input("Ridge alpha (λ)", value=1.0, min_value=0.0, step=0.1)
lasso_alpha = st.sidebar.number_input("Lasso alpha (λ)", value=0.1, min_value=0.0, step=0.01)
enet_alpha = st.sidebar.number_input("Elastic Net alpha (λ)", value=0.1, min_value=0.0, step=0.01)
enet_l1_ratio = st.sidebar.slider("Elastic Net l1_ratio (α)", 0.0, 1.0, 0.5, 0.05)

if not methods:
    st.warning("Select at least one method.")
    st.stop()


# ===================== 4. FIT MODELS =====================

models = {}
results = []

if "OLS" in methods:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["OLS"] = lr

if "Ridge" in methods:
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_train, y_train)
    models["Ridge"] = ridge

if "Lasso" in methods:
    lasso = Lasso(alpha=lasso_alpha)
    lasso.fit(X_train, y_train)
    models["Lasso"] = lasso

if "Elastic Net" in methods:
    enet = ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1_ratio)
    enet.fit(X_train, y_train)
    models["Elastic Net"] = enet

for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": name,
        "MSE": mse,
        "R²": r2
    })

metrics_df = pd.DataFrame(results)


# ===================== 5. TABS =====================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & Correlation",
    "📐 Formulas & Theory",
    "📋 Numerical Solution (Tables)",
    "📈 Plots (2D & 3D)",
    "📌 Likelihood & Priors (MLE vs MAP)",
    "🔔 Residuals & Gaussian"
])


# ========== TAB 1: Data & Correlation ==========
with tab1:
    st.subheader("Data summary")
    st.write(df.describe())

    st.subheader("Correlation matrix")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ========== TAB 2: Formulas & Theory ==========
with tab2:
    st.subheader("OLS (Ordinary Least Squares) – MLE")

    st.markdown(r"""
**Model:**

We assume a linear model with Gaussian noise:

\[
y = X\beta + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0,\sigma^2 I)
\]

**Log-likelihood:**

\[
\log p(y \mid X,\beta,\sigma^2) = -\frac{1}{2\sigma^2}\|y - X\beta\|^2 + C
\]

Maximizing this wrt \(\beta\) is same as minimizing the squared error:

\[
\hat{\beta}_{OLS} = \arg\min_\beta \|y - X\beta\|^2
\]

**Closed-form solution:**

\[
\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y
\]
""")

    st.subheader("Ridge Regression – MAP with Gaussian prior")

    st.markdown(r"""
Assume a Gaussian prior on coefficients:

\[
\beta \sim \mathcal{N}(0, \tau^2 I)
\]

Then the MAP estimate maximizes:

\[
\log p(\beta \mid X,y) = -\frac{1}{2\sigma^2}\|y - X\beta\|^2 -\frac{1}{2\tau^2}\|\beta\|^2 + C
\]

Equivalent minimization problem:

\[
\hat{\beta}_{Ridge} = \arg\min_\beta \left( \|y - X\beta\|^2 + \lambda \|\beta\|^2 \right)
\]

with \(\lambda = \sigma^2 / \tau^2\).

**Closed-form:**

\[
\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y
\]
""")

    st.subheader("Lasso – MAP with Laplace prior")

    st.markdown(r"""
Lasso can be seen as MAP with a Laplace (double-exponential) prior on \(\beta\).

\[
\hat{\beta}_{Lasso} = \arg\min_\beta \left( \|y - X\beta\|^2 + \lambda \|\beta\|_1 \right)
\]

No simple closed-form solution; solved by optimization algorithms.
""")

    st.subheader("Elastic Net")

    st.markdown(r"""
Elastic Net combines L1 and L2 penalties:

\[
\hat{\beta}_{EN} = \arg\min_\beta \left(
\|y - X\beta\|^2 + \lambda \big( \alpha \|\beta\|_1 + (1-\alpha)\|\beta\|_2^2 \big)
\right)
\]

- \(\alpha = 1\): Lasso
- \(\alpha = 0\): Ridge
""")


# ========== TAB 3: Numerical Solution (Tables) ==========
with tab3:
    st.subheader("Model performance metrics")
    st.dataframe(metrics_df.style.format({"MSE": "{:.4f}", "R²": "{:.4f}"}))

    st.subheader("Coefficient tables")

    for name, model in models.items():
        st.markdown(f"### {name}")

        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": model.coef_
        })
        st.write("Intercept:", model.intercept_)
        st.dataframe(coef_df)

    # Step-by-step OLS numeric (using training data)
    if "OLS" in models and X_train.shape[1] <= 5:
        st.subheader("OLS step-by-step numeric (on training data)")

        X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        XT_X = X_design.T @ X_design
        XT_y = X_design.T @ y_train
        try:
            XT_X_inv = np.linalg.inv(XT_X)
        except np.linalg.LinAlgError:
            XT_X_inv = np.linalg.pinv(XT_X)
            st.warning("X^T X is singular; using pseudo-inverse for OLS step-by-step numeric.")
        beta_hat = XT_X_inv @ XT_y

        st.markdown("Design matrix with bias term (first column = 1):")
        st.write(pd.DataFrame(X_design, columns=["1 (bias)"] + feature_cols))

        st.markdown(r"$X^T X$:")
        st.write(pd.DataFrame(XT_X))

        st.markdown(r"$(X^T X)^{-1}$:")
        st.write(pd.DataFrame(XT_X_inv))

        st.markdown(r"$X^T y$:")
        st.write(pd.DataFrame(XT_y, columns=["X^T y"]))

        st.markdown(r"**β̂ (OLS coefficients including intercept)**:")
        beta_df = pd.DataFrame({
            "Parameter": ["Intercept"] + feature_cols,
            "Value": beta_hat
        })
        st.dataframe(beta_df)
    else:
        st.info("For step-by-step OLS numeric, use OLS and ≤ 5 features.")


# ========== TAB 4: Plots (2D & 3D) ==========
with tab4:
    st.subheader("2D & 3D Visualizations")

    # 1D scatter + line (only if one feature selected)
    if len(feature_cols) == 1:
        x_col = feature_cols[0]
        st.markdown(f"### 2D scatter & regression line ({x_col} → {target_col})")

        fig, ax = plt.subplots()
        ax.scatter(X_train[:, 0], y_train, label="Train", alpha=0.7)
        colors = ["red", "green", "purple", "orange"]
        color_idx = 0

        x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)

        for name, model in models.items():
            y_line = model.predict(x_line)
            ax.plot(x_line, y_line, label=name)  # default colors

        ax.set_xlabel(x_col)
        ax.set_ylabel(target_col)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Select exactly 1 feature to see 2D scatter + line.")

    # 3D plot (if at least 2 features)
    if X.shape[1] >= 2:
        st.markdown("### 3D surface plot (using first 2 features)")

        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_name = feature_cols[0]
        x2_name = feature_cols[1]

        # Use OLS if available, else first model
        model_for_3d = models.get("OLS", list(models.values())[0])

        grid_x1 = np.linspace(x1.min(), x1.max(), 30)
        grid_x2 = np.linspace(x2.min(), x2.max(), 30)
        gx1, gx2 = np.meshgrid(grid_x1, grid_x2)
        grid_points = np.c_[gx1.ravel(), gx2.ravel()]

        # If more than 2 features, pad others with mean
        if X.shape[1] > 2:
            extra_means = X[:, 2:].mean(axis=0)
            extra = np.tile(extra_means, (grid_points.shape[0], 1))
            full_grid = np.hstack([grid_points, extra])
        else:
            full_grid = grid_points

        gy = model_for_3d.predict(full_grid).reshape(gx1.shape)

        fig3d = px.scatter_3d(
            x=x1, y=x2, z=y,
            labels={"x": x1_name, "y": x2_name, "z": target_col},
            opacity=0.6,
            title="Data points (3D)"
        )
        st.plotly_chart(fig3d, use_container_width=True)

        surface3d = px.surface(
            x=grid_x1, y=grid_x2, z=gy,
            labels={"x": x1_name, "y": x2_name, "z": target_col},
            title=f"Regression surface (model: {list(models.keys())[0]})"
        )
        st.plotly_chart(surface3d, use_container_width=True)


# ========== TAB 5: Likelihood & Priors (MLE vs MAP) ==========
with tab5:
    st.subheader("Likelihood (Gaussian noise assumption)")

    st.markdown(r"""
Assume residuals \( \varepsilon = y - X\beta \) are Gaussian:

\[
\varepsilon \sim \mathcal{N}(0,\sigma^2 I)
\]

Then

\[
p(y \mid X,\beta) 
= \prod_{i} \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left( -\frac{(y_i - x_i^T \beta)^2}{2\sigma^2} \right)
\]

and the log-likelihood is proportional to **negative squared error**.
""")

    if "OLS" in models:
        model = models["OLS"]
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        sigma2_hat = np.mean(residuals**2)

        st.write("Estimated noise variance (σ²) from OLS residuals:", sigma2_hat)

        log_like = -0.5 * np.sum((residuals**2) / sigma2_hat + np.log(2 * np.pi * sigma2_hat))
        st.write("Approximate log-likelihood (train, OLS):", log_like)
    else:
        st.info("Enable OLS to see numeric log-likelihood.")

    st.subheader("MAP: Ridge, Lasso, Elastic Net as Penalized Likelihood")

    st.markdown(r"""
- **Ridge** = MLE with **Gaussian prior** on \(\beta\)  
- **Lasso** = MLE with **Laplace prior** on \(\beta\)  
- **Elastic Net** = combination of L1 + L2 priors

MAP objective:

\[
\hat{\beta}_{MAP} =
\arg\max_\beta \log p(y \mid X,\beta) + \log p(\beta)
\]

Which corresponds to

\[
\arg\min_\beta \left(
\|y - X\beta\|^2 + \text{penalty}(\beta)
\right)
\]
""")


# ========== TAB 6: Residuals & Gaussian ==========
with tab6:
    st.subheader("Residual analysis")

    chosen_model_name = st.selectbox("Choose model for residual analysis", list(models.keys()))
    chosen_model = models[chosen_model_name]

    y_pred_train = chosen_model.predict(X_train)
    residuals = y_train - y_pred_train

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Histogram of residuals** (with Gaussian curve)")
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_xlabel("Residual")
        st.pyplot(fig)

    with col2:
        st.markdown("**Residuals vs Fitted**")
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred_train, residuals, alpha=0.7)
        ax2.axhline(0, linestyle="--")
        ax2.set_xlabel("Fitted values")
        ax2.set_ylabel("Residuals")
        st.pyplot(fig2)

    st.markdown("""
If residuals are roughly Gaussian and centered around zero, the linear model and noise assumption are more reasonable.
""")
