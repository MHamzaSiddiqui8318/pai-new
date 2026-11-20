# Streamlit Deployment Guide
## AI-Based Patient Admission Prediction System

### 🚀 Option 1: Streamlit Community Cloud (FREE — Recommended)

#### Step 1: GitHub pe push karo
```bash
cd "d:\New folder\PAI-CCA-main\PAI-CCA-main"
git init
git add .
git commit -m "Add Streamlit web app"
git remote add origin https://github.com/YOUR_USERNAME/PAI-CCA.git
git push -u origin main
```

#### Step 2: Streamlit Cloud pe deploy karo
1. Jao **[share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. Click **"New app"**
4. Select your repo: `YOUR_USERNAME/PAI-CCA`
5. Branch: `main`
6. Main file path: `streamlit_app.py`
7. Click **"Deploy!"**

🎉 **Done!** Aapko ek public URL milega jaise:
`https://your-username-pai-cca-streamlit-app-xxxxx.streamlit.app`

---

### 🖥️ Option 2: Locally Run Karna Ho

```bash
pip install streamlit
cd "d:\New folder\PAI-CCA-main\PAI-CCA-main"
streamlit run streamlit_app.py
```
Browser mein automatically `http://localhost:8501` khulega.

---

### ⚠️ Important Notes
- `outputs/models/admission_model.joblib` file (~603 KB) GitHub pe push honi chahiye
- `outputs/models/model_info.json` aur `demo_patients.csv` bhi zaroori hain
- `outputs/figures/` aur `outputs/metrics/` folders bhi push karo (charts ke liye)
- `.gitignore` mein yeh files block nahi honi chahiye

