# streamlit_app/app.py
import os
import sys
import io
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Make ml package importable when deployed (Streamlit Cloud / Docker)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import your ml helpers (they should only define functions)
from ml.preprocess import load_and_preprocess
from ml.train import get_models
from ml.evaluate import evaluate_models
from ml.explain import explain_model
from ml.insights import generate_insights
from ml.pdfgen import generate_pdf

st.set_page_config(page_title="InsightX Dashboard", layout="wide")

# -----------------------
# Particle background (tsParticles)
# -----------------------
TRANSPARENT_CSS = """
<style>
/* Transparent app backgrounds so particle canvas shows */
[data-testid="stAppViewContainer"] > .main { background: transparent; }
[data-testid="stAppViewContainer"] { background: transparent; }

/* Place the tsParticles canvas behind the app */
#tsparticles { position: fixed; top:0; left:0; width:100%; height:100%; z-index: -1; pointer-events: none; }
</style>
"""
st.markdown(TRANSPARENT_CSS, unsafe_allow_html=True)

def inject_background_particles():
    particle_html = r"""
    <div id="tsparticles"></div>
    <script src="https://cdn.jsdelivr.net/npm/tsparticles@2.8.0/tsparticles.bundle.min.js"></script>
    <script>
    (async () => {
      await tsParticles.load("tsparticles", {
        fpsLimit: 60,
        background: { color: { value: "transparent" } },
        particles: {
          number: { value: 50, density: { enable: true, area: 800 } },
          color: { value: ["#24d9ff", "#9b7bff", "#5fffb0"] },
          shape: { type: "circle" },
          opacity: { value: 0.7, random: { enable: true, minimumValue: 0.3 } },
          size: { value: { min: 1, max: 4 }, random: true },
          links: { enable: true, distance: 120, color: "#2b2b2b", opacity: 0.25, width: 1 },
          move: { enable: true, speed: 0.8, random: true, outModes: { default: "out" } }
        },
        interactivity: {
          detectsOn: "canvas",
          events: { onHover: { enable: true, mode: "repulse" }, onClick: { enable: false }, resize: true },
          modes: { repulse: { distance: 100, duration: 0.4 } }
        },
        detectRetina: true
      });
    })();
    </script>
    """
    components.html(particle_html, height=1, scrolling=False)

def particle_burst():
    burst_html = """
    <canvas id="burstCanvas" style="position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0;"></canvas>
    <script>
    (function(){
      const canvas = document.getElementById('burstCanvas');
      const ctx = canvas.getContext('2d');
      function resize(){ canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
      window.addEventListener('resize', resize); resize();

      let particles = [];
      const colors = ["#5fffb0","#9b7bff","#24d9ff"];
      function createBurst(){
        particles = [];
        const cx = canvas.width * 0.5;
        const cy = 150; // from top (so it appears near header)
        for(let i=0;i<80;i++){
          particles.push({
            x: cx, y: cy,
            vx: (Math.random()-0.5)*8,
            vy: (Math.random()-0.5)*8 - 2,
            r: 2 + Math.random()*3,
            alpha: 1,
            color: colors[Math.floor(Math.random()*colors.length)]
          });
        }
        animate();
      }
      function animate(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        for(let i=0;i<particles.length;i++){
          const p = particles[i];
          p.vy += 0.1; // gravity feel
          p.x += p.vx; p.y += p.vy; p.alpha -= 0.015;
          ctx.globalAlpha = Math.max(p.alpha,0);
          ctx.fillStyle = p.color;
          ctx.beginPath();
          ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
          ctx.fill();
        }
        particles = particles.filter(p => p.alpha > 0);
        if(particles.length>0) requestAnimationFrame(animate);
      }
      // expose createBurst to window so we can call it via Streamlit HTML re-eval
      window.createInsightBurst = createBurst;
      // auto-run on injection
      createBurst();
    })();
    </script>
    """
    # height 1 so iframe doesn't take space - canvas is fixed
    components.html(burst_html, height=1, scrolling=False)

# inject the background particles once
inject_background_particles()

# -----------------------
# Header
# -----------------------
st.title("📊 InsightX – ML Visualization Dashboard")
st.write("Upload a dataset → Explore → Train ML models → Get Insights")

# -----------------------
# Upload section
# -----------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])

# Only trigger burst and load data after upload exists
if uploaded_file is not None:
    # show a burst animation (non-blocking)
    particle_burst()

    # Attempt to read file robustly
    try:
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
    except Exception:
        # fallback in case of binary bytes / other encodings
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())

else:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# store raw df in session state for reuse
st.session_state["df"] = df

# -----------------------
# Particle Scatter (data-driven) - WebGL accelerated
# -----------------------
st.subheader("✨ Particle Data Visualization")
def show_particle_scatter(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Upload a dataset with at least 2 numeric columns for particle visualization.")
        return

    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        opacity=0.8,
        color_continuous_scale="Plasma",
        render_mode="webgl",
        title=f"Data Particle Visualization — {x_col} vs {y_col}"
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

show_particle_scatter(df)

# -----------------------
# Exploratory Data Analysis
# -----------------------
st.subheader("🔎 Exploratory Data Analysis")
st.write("### Dataset Info")
st.write(df.describe(include='all'))

st.write("### Missing values")
missing = df.isnull().sum()
st.table(missing[missing > 0])

# Correlation heatmap (numeric only)
try:
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not draw heatmap: {e}")

# -----------------------
# Interactive Visualizations
# -----------------------
st.subheader("📈 Interactive Visualizations")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(numeric_cols) >= 2:
    x = st.selectbox("Select X-axis", numeric_cols, index=0)
    y = st.selectbox("Select Y-axis", numeric_cols, index=1)
    fig = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("At least two numeric columns required for custom interactive plots.")

# -----------------------
# ML Pipeline
# -----------------------
st.subheader("🤖 Train ML Models")

# Run pipeline only when user clicks
if st.button("Run ML Pipeline"):
    st.info("Training models... (this may take a little while)")

    # preprocess function expects either a path or a file-like object (your preprocess should handle both)
    try:
        (X_train, X_test, y_train, y_test), le = load_and_preprocess(uploaded_file)
    except ValueError as ve:
        st.error(str(ve))
        st.stop()
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    models = get_models()
    try:
        results = evaluate_models(models, X_train, X_test, y_train, y_test)
    except Exception as e:
        st.error(f"Model training/evaluation failed: {e}")
        st.stop()

    st.write("### 📊 Model Performance")
    st.dataframe(results)

    # pick best model
    best_row = results.sort_values(by="Accuracy", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = models[best_name]
    # save best model file
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_streamlit_model.pkl")

    # store in session state for further tabs
    st.session_state["best_model"] = best_model
    st.session_state["label_encoder"] = le
    st.session_state["X_test"] = X_test
    st.session_state["results"] = results

    # Generate auto insights and store
    try:
        auto_report = generate_insights(df, results)
    except Exception as e:
        auto_report = f"Auto-insights generation failed: {e}"
    st.session_state["auto_report"] = auto_report

    st.success(f"Training finished. Best model: {best_name}")

# -----------------------
# Predictions (uses model from session_state)
# -----------------------
st.subheader("🔮 Make Predictions (example)")
if st.button("Predict on New Sample"):
    if "best_model" not in st.session_state:
        st.warning("Train the models first using 'Run ML Pipeline'.")
    else:
        best_model = st.session_state["best_model"]
        le = st.session_state["label_encoder"]
        X_test = st.session_state["X_test"]
        # simple sample
        sample = X_test.iloc[0:1]
        pred = best_model.predict(sample)[0]
        try:
            label = le.inverse_transform([pred])[0]
        except Exception:
            label = str(pred)
        st.info(f"Prediction: **{label}**")
        st.write("Input features used:")
        st.dataframe(sample)

# -----------------------
# Explainability (SHAP)
# -----------------------
st.subheader("🔍 Explainability (SHAP)")

if st.button("Generate SHAP Summary"):
    if "best_model" not in st.session_state:
        st.warning("Train the models first.")
    else:
        model = st.session_state["best_model"]
        X_test = st.session_state["X_test"]
        # use a small sample to speed up
        X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
        st.info("Generating SHAP plot (may take a moment)...")
        try:
            fig = explain_model(model, X_sample)
            st.pyplot(fig)
            st.success("SHAP plot generated.")
        except Exception as e:
            st.error(f"SHAP explainability failed: {e}")

# -----------------------
# Auto Insights & PDF download
# -----------------------
st.subheader("📄 Auto Insights & Report")

if "auto_report" in st.session_state:
    st.text_area("📜 Auto-Generated Insights", st.session_state["auto_report"], height=220)
else:
    st.info("Run ML Pipeline to generate auto-insights (and then the PDF).")

if "auto_report" in st.session_state:
    if st.button("Download PDF Report"):
        try:
            path = generate_pdf(st.session_state["auto_report"])
            with open(path, "rb") as f:
                st.download_button("⬇ Download PDF", data=f, file_name="InsightX_Report.pdf")
            st.success("PDF ready for download.")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")

st.subheader("📄 Generate PDF Report")

if 'auto_report' not in st.session_state:
    st.warning("Please run the ML pipeline first to generate insights.")
else:
    if st.button("Download Report"):
        path = generate_pdf(st.session_state["auto_report"])
        with open(path, "rb") as f:
            st.download_button(
                "⬇ Download PDF",
                data=f,
                file_name="InsightX_Report.pdf"
            )

# -----------------------
# Footer / tips
# -----------------------
st.markdown("---")
st.write("Tip: Use datasets with a `type` column for classification examples (or update your preprocess.py to change the target).")