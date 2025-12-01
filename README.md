<h1 align="center">🧠 InsightX – AI-Powered ML Visualization Dashboard</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-brightgreen?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/License-MIT-purple" />
</p>

<p align="center">
  <b>An interactive data exploration + machine learning dashboard built with Streamlit.</b><br/>
  Upload any dataset → Explore → Train ML models → Generate insights → Explain predictions → Export PDF reports.
</p>

---

## 🔗 **Live Demo**
👉 **Streamlit App:** *(https://lelixn-insightx.streamlit.app/)*  

---
📸 UI Preview

Below is an example of the InsightX interface while uploading and visualizing a dataset:

Upload → Validate → Preview → Explore → Generate ML Insights

<img width="1844" height="877" alt="Screenshot 2025-12-01 230240" src="https://github.com/user-attachments/assets/00a3d872-04ab-40ef-a082-39df94915b6d" />


# 🚀 Features

### 🧩 **1. Dataset Upload**
- Upload any CSV file
- Auto preview & validation

### 📊 **2. Automated EDA**
- Summary statistics  
- Correlation heatmaps  
- Missing value detection  
- Distribution plots  

### 📈 **3. Interactive Visualizations**
- Plotly-powered charts  
- Select X/Y axes live  
- Interactive scatter, bar, distributions  

### 🤖 **4. ML Training Pipeline**
- Clean preprocessing (categorical encoding, date parsing, duration cleaning)
- Multiple ML models:
  - Logistic Regression  
  - Random Forest  
  - KNN  
- Auto model comparison table  
- Saves best model

### 🔍 **5. Explainability with SHAP**
- Global feature importance  
- SHAP summary plot  
- Interpret why predictions happen

### 🧠 **6. Auto Insights Generator**
- Dataset insights  
- Correlation observations  
- Best model report  

### 📄 **7. PDF Report Generator**
- Auto-generated report summarizing dataset + ML results  
- One-click download  

---

# 🛠️ Installation & Local Run

###  Clone the repository  
```bash
git clone https://github.com/lelixn/InsightX.git
cd InsightX
```
Install dependencies
```
pip install -r requirements.txt
```
Run the Streamlit dashboard
```
python -m streamlit run streamlit_app/app.py
```

☁️ Deployment (Streamlit Cloud)
```
Push code to GitHub

Go to https://streamlit.io/cloud
Create new app
Repo: lelixn/InsightX
Path: streamlit_app/app.py
Deploy 🚀
```

If you like this project, consider starring ⭐ the repo! <br>
Made with ❤️ by Lelien Panda.
