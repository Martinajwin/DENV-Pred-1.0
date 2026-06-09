# ==========================================================
# 🧬 DENV-Pred 1.0 (Pan-Serotype SBML Server)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
import joblib
from graphviz import Digraph

warnings.filterwarnings("ignore")

# -----------------------------
# 🎨 Custom Colorful CSS Injector
# -----------------------------
st.set_page_config(page_title="DENV-Pred 1.0", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    /* Vibrant Headers */
    h1 { color: #2E86C1 !important; text-align: center; }
    h2 { color: #8E44AD !important; }
    h3 { color: #28B463 !important; text-align: center; margin-top: -15px; }
    
    /* Styled Execute Button */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #FF4B2B, #FF416C);
        color: white;
        font-weight: bold;
        font-size: 18px;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(to right, #FF416C, #FF4B2B);
        transform: scale(1.02);
    }
    
    /* Highlighted metric boxes */
    .metric-box {
        background-color: #E8F8F5;
        border-left: 5px solid #1ABC9C;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🎯 Receptor Constraints Library
# -----------------------------
TARGET_CONSTRAINTS = {
    "DENV-1 NS3 Protease": {"Req_HBA": 2, "Req_HBD": 1, "Req_Aro": 1, "Vol": 11200},
    "DENV-2 NS3 Protease": {"Req_HBA": 2, "Req_HBD": 1, "Req_Aro": 1, "Vol": 10078},
    "DENV-3 NS3 Protease": {"Req_HBA": 3, "Req_HBD": 1, "Req_Aro": 1, "Vol": 10500},
    "DENV-4 NS3 Protease": {"Req_HBA": 2, "Req_HBD": 2, "Req_Aro": 0, "Vol": 10800},
    "DENV-1 NS5 Polymerase": {"Req_HBA": 3, "Req_HBD": 0, "Req_Aro": 2, "Vol": 25000},
    "DENV-2 NS5 Polymerase": {"Req_HBA": 2, "Req_HBD": 0, "Req_Aro": 2, "Vol": 26485},
    "DENV-3 NS5 Polymerase": {"Req_HBA": 3, "Req_HBD": 1, "Req_Aro": 2, "Vol": 25800},
    "DENV-4 NS5 Polymerase": {"Req_HBA": 2, "Req_HBD": 0, "Req_Aro": 1, "Vol": 26100}
}

# -----------------------------
# ⚙️ Pipeline Asset Loader
# -----------------------------
@st.cache_resource(show_spinner="Loading DENV-Pred 1.0 Framework...")
def load_pipeline_assets():
    try:
        vt = joblib.load("VarianceThreshold_Filter.pkl")
        selector = joblib.load("Feature_Selector.pkl")
        ad_model = joblib.load("DENV_Applicability_Domain.pkl")
        classifier = joblib.load("GradientBoosting_Classifier_Tuned.pkl")
        regressor = joblib.load("GB_Reg_Regressor_Tuned.pkl")
        return vt, selector, ad_model, classifier, regressor
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}. Ensure all 5 .pkl files are in the directory.")
        return None, None, None, None, None

def extract_advanced_sbml(smiles, target_reqs):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    ligand_hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    ligand_hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    ligand_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    
    hbd_match = 1 if ligand_hbd >= target_reqs["Req_HBD"] else 0
    hba_match = 1 if ligand_hba >= target_reqs["Req_HBA"] else 0
    aromatic_match = 1 if ligand_aromatic_rings >= target_reqs["Req_Aro"] else 0
    steric_clash = 1 if (mw * 1.5) > target_reqs["Vol"] else 0 
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.zeros((1,))
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, fp_array)
    
    sbml_features = [mw, logp, ligand_hbd, ligand_hba, ligand_aromatic_rings, hbd_match, hba_match, aromatic_match, steric_clash]
    return np.concatenate((fp_array, sbml_features)), sbml_features

# -----------------------------
# ⚗️ Streamlit UI Layout
# -----------------------------
st.title("🧬 Pan-Serotype Dengue Inhibitor Predictor")  
st.markdown("### 🔬 (DENV-Pred 1.0)")

tabs = st.tabs(["1️⃣ Virtual Screening", "2️⃣ Methodology", "3️⃣ Model Performance", "4️⃣ Intellectual Property"])
tab1, tab2, tab3, tab4 = tabs

# ==========================================================
# 1️⃣ SCREENING TAB (MAIN SCREEN REWRITE)
# ==========================================================
with tab1:
    vt, selector, ad_model, classifier, regressor = load_pipeline_assets()

    st.markdown("<div class='metric-box' style='color: #000000;'><b>Welcome to DENV-Pred 1.0:</b> A high-stringency thermodynamic gatekeeper for early-stage Pan-Flavivirus drug discovery.</div>", unsafe_allow_html=True)
    
    # --- BEAUTIFUL MULTI-COLUMN INPUT UI ---
    st.markdown("### ⚙️ Configure Screening Parameters")
    
    col_settings, col_input = st.columns([1, 1.2], gap="large")
    
    with col_settings:
        st.info("🎯 **Target Selection**")
        selected_target = st.selectbox("Select Viral Target constraints:", list(TARGET_CONSTRAINTS.keys()))
        
        st.warning("🎛️ **Model Stringency (Diagnostic Tool)**")
        st.markdown("*Lowering this captures weak binders but increases false positives.*")
        decision_threshold = st.slider("Classification Threshold", 0.40, 0.80, 0.50, 0.05)

    with col_input:
        st.success("🧪 **Input Molecules**")
        input_option = st.radio("Input Type:", ["Enter SMILES manually", "Upload CSV"], horizontal=True)
        
        smiles_list = []
        if input_option == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with 'SMILES' column", type=["csv"])
            if uploaded_file is not None:
                df_input = pd.read_csv(uploaded_file)
                if "SMILES" not in df_input.columns:
                    st.error("CSV must contain a 'SMILES' column.")
                else:
                    smiles_list = [str(s) for s in df_input["SMILES"] if pd.notna(s)]
        else:
            user_smiles = st.text_area("Enter SMILES (one per line):", "CC1=CC(=CC=C1)S(=O)(=O)NC2=CC=C(C=C2)C(=O)O\nCC(=O)OC1=CC=CC=C1C(=O)O", height=130)
            smiles_list = [s.strip() for s in user_smiles.split("\n") if s.strip()]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered Execute Button
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        execute_btn = st.button("🚀 EXECUTE SBML PIPELINE", use_container_width=True)

    # --- MAIN SCREEN RESULTS WITH DIAGNOSTICS ---
    if execute_btn and smiles_list and classifier is not None:
        st.markdown("---")
        st.info(f"🧬 Applying geometric pocket constraints for **{selected_target}**... please wait ⏳")
        target_reqs = TARGET_CONSTRAINTS[selected_target]
        results_data = []

        for i, smi in enumerate(smiles_list):
            extraction = extract_advanced_sbml(smi, target_reqs)
            
            if extraction is None:
                results_data.append({"SMILES": smi, "Status": "⛔ INVALID", "AD_Check": "-", "Class": "-", "Confidence": "-", "pIC50": "-", "IC50_nM / Diagnostics": "RDKit Parsing Failed"})
                continue
                
            raw_features, physics_features = extraction
            
            raw_vector = raw_features.reshape(1, -1)
            vector_vt = vt.transform(raw_vector)
            final_vector = selector.transform(vector_vt)
            
            # 1. Applicability Domain
            in_domain = ad_model.predict(final_vector)[0]
            if in_domain == -1:
                results_data.append({"SMILES": smi, "Status": "⚠️ OUT OF DOMAIN", "AD_Check": "Fail", "Class": "Skipped", "Confidence": "-", "pIC50": "-", "IC50_nM / Diagnostics": "Alien Structure Rejected"})
                continue
                
            # 2. Gatekeeper Classifier (WITH DIAGNOSTIC LOGIC)
            confidence = classifier.predict_proba(final_vector)[0][1]
            is_active = 1 if confidence >= decision_threshold else 0
            
            fail_reasons = []
            if physics_features[5] == 0: fail_reasons.append("Failed HBD")
            if physics_features[6] == 0: fail_reasons.append("Failed HBA")
            if physics_features[7] == 0: fail_reasons.append("Failed Aromatic")
            if physics_features[8] == 1: fail_reasons.append("Steric Clash (Too Large)")
            
            if is_active == 0:
                reason_str = " | ".join(fail_reasons) if fail_reasons else "Low Model Probability"
                results_data.append({"SMILES": smi, "Status": "❌ REJECTED", "AD_Check": "Pass", "Class": "Inactive", "Confidence": f"{(1-confidence):.1%}", "pIC50": "-", "IC50_nM / Diagnostics": f"Reason: {reason_str}"})
                continue
                
            # 3. Ranker Regressor
            pred_pic50 = regressor.predict(final_vector)[0]
            pred_ic50_nm = 10 ** (9 - pred_pic50)
            
            results_data.append({"SMILES": smi, "Status": "✅ SUCCESS", "AD_Check": "Pass", "Class": "ACTIVE HIT", "Confidence": f"{confidence:.1%}", "pIC50": round(pred_pic50, 2), "IC50_nM / Diagnostics": f"{round(pred_ic50_nm, 2)} nM"})

        st.subheader("📊 Screening Results & Diagnostics")
        
        # Color styling for dataframe
        results_df = pd.DataFrame(results_data)
        def color_status(val):
            if 'SUCCESS' in str(val) or 'ACTIVE' in str(val): return 'color: #27AE60; font-weight: bold;'
            if 'REJECTED' in str(val) or 'Inactive' in str(val): return 'color: #C0392B;'
            if 'DOMAIN' in str(val) or 'Skipped' in str(val): return 'color: #F39C12;'
            return ''
            
        styled_df = results_df.style.map(color_status, subset=['Status', 'Class'])
        st.dataframe(styled_df, use_container_width=True, height=250)
        
        csv_standard = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Diagnostic Report", data=csv_standard, file_name=f"{selected_target}_diagnostics.csv", mime="text/csv")


# ==========================================================
# 2️⃣ METHODOLOGY TAB
# ==========================================================
with tab2:
    st.header("Structure-Based Machine Learning (SBML) Workflow")

    st.markdown("""
     <div class='metric-box' style='background-color: #EBF5FB; border-left: 5px solid #2874A6; color: #000000;'>
     <b>Dual-Architecture Pipeline for Pan-Serotype Compound Screening</b><br>
     Traditional virtual screening often suffers from the Curse of Dimensionality and extreme false-positive rates due to the reliance on pure 2D cheminformatics. DENV-Pred 1.0 implements an advanced <b>Structure-Based Machine Learning (SBML)</b> framework.
     </div>
     """, unsafe_allow_html=True)
    
    st.write("""
    Input SMILES are transformed into 1,024-bit Morgan Fingerprints combined with 9 explicit physical parameters (Molecular Weight, LogP, and strict Hydrogen Bond/Steric complementarity vectors dynamically mapped to specific Dengue viral pockets). 

    To prevent model overfitting, the 1,033-dimensional space undergoes intense feature reduction via `VarianceThreshold` and a Random Forest `SelectFromModel`. The imbalanced chemical space (1:90 actives to decoys) is handled during training via **SMOTEENN**, which oversamples active hits while mathematically cleaning noisy decision boundaries. 

    **Inference is protected by a 3-Stage hierarchical protocol:**
    1.  **Applicability Domain (AD):** An Isolation Forest evaluates structural novelty.
    2.  **The Gatekeeper:** A tuned Gradient Boosting Classifier, optimized strictly for F1-Score, identifies the hit compound.
    3.  **The Ranker:** A Gradient Boosting Regressor predicts the exact thermodynamic binding affinity ($pIC_{50}$) for lead prioritization.
    """)

    st.subheader("DENV-Pred 1.0 Architecture")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        dot = Digraph("SBMLFlow", engine="dot")
        dot.attr(rankdir="TB", splines="true", nodesep="0.5", ranksep="0.6")
        dot.attr("node", shape="box", style="rounded,filled,solid", fontsize="11", margin="0.2,0.15", fontname="Arial")
        
        dot.node("Start", "<<b>USER INPUT</b><br align='left'/>SMILES + Target Selection>", fillcolor="#FCF3CF")
        dot.node("Extract", "<<b>SBML EXTRACTION</b><br align='left'/>• 1024-Bit Fingerprint<br align='left'/>• Geometric Pocket Matching (HBD/HBA/Steric)>", fillcolor="#D4E6F1")
        dot.node("Reduce", "<<b>DIMENSIONALITY REDUCTION</b><br align='left'/>• VarianceThreshold Filter<br align='left'/>• SelectFromModel (Top Features)>", fillcolor="#E8DAEF")
        
        dot.node("AD", "<<b>STAGE 1: ISOLATION FOREST (AD)</b><br align='left'/>Check structural familiarity>", fillcolor="#F5B7B1")
        dot.node("Gatekeeper", "<<b>STAGE 2: GB CLASSIFIER</b><br align='left'/>Optimized via SMOTEENN &amp; F1-Score>", fillcolor="#AED6F1")
        dot.node("Ranker", "<<b>STAGE 3: GB REGRESSOR</b><br align='left'/>Compute exact pIC50 Affinity>", fillcolor="#A9DFBF")

        with dot.subgraph() as s_out:
            s_out.attr(rank='same')
            s_out.node("Out", "<<b>OUT OF DOMAIN</b><br align='left'/>Rejected>", fillcolor="#E5E7E9")
            s_out.node("Inactive", "<<b>INACTIVE</b><br align='left'/>Rejected>", fillcolor="#E5E7E9")
            s_out.node("Active", "<<b>PRIORITY HIT</b><br align='left'/>IC50 Predicted>", fillcolor="#F9E79F")

        dot.edge("Start", "Extract")
        dot.edge("Extract", "Reduce")
        dot.edge("Reduce", "AD")
        dot.edge("AD", "Out", label=" Fail", fontcolor="red")
        dot.edge("AD", "Gatekeeper", label=" Pass", fontcolor="green")
        dot.edge("Gatekeeper", "Inactive", label=" Class 0", fontcolor="red")
        dot.edge("Gatekeeper", "Ranker", label=" Class 1", fontcolor="green")
        dot.edge("Ranker", "Active")

        st.graphviz_chart(dot, use_container_width=True)


# ==========================================================
# 3️⃣ MODEL PERFORMANCE TAB
# ==========================================================
with tab3:
    st.header("Validation & Performance Metrics")

    st.write("""
        DENV-Pred 1.0 is constructed on a rigorously filtered dataset of verified Pan-Dengue (DENV 1-4) nanomolar inhibitors ($pIC_{50} \ge 6.0$) sourced from the ChEMBL database, pitted against thousands of inactive decoys. All final metrics are reported using **95% Confidence Intervals (CI)** derived from 1,000 bootstrap iterations via the bias-corrected and accelerated (BCa) method.
    """)

    st.subheader("The Reality of High-Stringency Screening (40x Enrichment)")
    st.info("""
    **Understanding the Evaluation Context:** The training data features a severe **1:90 class imbalance** (~1% Actives vs 99% Inactives). Randomly sampling chemical space yields a 1% hit rate. 
    
    The tuned Gradient Boosting Gatekeeper achieves a **Precision of ~41%** and a **Recall of 87.5%**. This means the model successfully captures nearly 90% of all true Dengue inhibitors, while boosting the discovery hit-rate by **40 times over random chance**. This punitive framework deliberately prioritizes absolute confidence over baseline accuracy.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Gatekeeper: Gradient Boosting Classifier**")
        class_data = {
            "Metric": ["Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC", "PR-AUC"],
            "Value": ["0.4118", "0.8750", "0.5600", "0.9953", "0.7831"]
        }
        st.table(pd.DataFrame(class_data))
        st.markdown("*Bootstrapped F1-Score 95% CI: [0.286 - 0.774]*")
        
    with col2:
        st.success("**Ranker: Gradient Boosting Regressor**")
        reg_data = {
            "Metric": ["R² Score", "MSE", "RMSE", "MAE"],
            "Value": ["0.4235", "0.0812", "0.2849", "0.1715"]
        }
        st.table(pd.DataFrame(reg_data))
        st.markdown("*Bootstrapped R² 95% CI: [0.169 - 0.629]*")


# ==========================================================
# 4️⃣ REFERENCES & IP TAB
# ==========================================================
with tab4:
    st.header("Intellectual Property & Resources")
    
    st.markdown("### Institutional Affiliation & Copyright")
    st.markdown("**© 2026 Manipal Academy of Higher Education (MAHE). All rights reserved.**")
    st.markdown("Developed by: **D. Kumar, A. J. Martin**")
    st.markdown("*The algorithms, consensus logic, and trained models associated with DENV-Pred 1.0 are the intellectual property of Manipal Academy of Higher Education (MAHE).*")
    st.markdown("---")
    
    st.markdown("### How to Cite DENV-Pred 1.0 (Webtool Citation)")
    st.markdown("If you use the DENV-Pred webtool in research or publications, please cite:")
    
    st.info("**DENV-Pred 1.0 Webtool** | D. Kumar, A. J. Martin | Manipal Academy of Higher Education (MAHE) | Version 1.0 (2026).")

    st.markdown("---")

    st.markdown("""
### Scientific Literature & Computational Packages
#### 1. Machine Learning & Over-sampling
* **Lundberg, S. M., & Lee, S. I.** A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS* (2017).
* **Batista, G. E., et al.** A study of the behavior of several methods for balancing machine learning training data (SMOTEENN). *SIGKDD Explor. Newsl.* (2004).
* **Pedregosa et al.** Scikit-Learn: Machine Learning in Python. *JMLR* 12, 2825–2830 (2011).

#### 2. Descriptor Generation & Cheminformatics
* **RDKit:** Open-source cheminformatics. [http://www.rdkit.org](http://www.rdkit.org).
* **Morgan, H. L.** The Generation of a Unique Machine Description for Chemical Structures (ECFP Fingerprints). *J. Chem. Doc.* (1965).

#### 3. Software, Platforms & Versions (Used in DENV-Pred 1.0)
| Software / Package | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10 | Core Development |
| **Streamlit** | 1.50 | Web Interface Deployment |
| **RDKit** | 2025.03.6 | Cheminformatics and Fingerprinting |
| **scikit-learn** | 1.4.2 | Feature Selection, Isolation Forest, GB Models |
| **imbalanced-learn** | 0.12.2 | SMOTEENN Data Balancing |
| **SHAP** | latest | Model Interpretability & SAR |
| **NumPy / Pandas** | 1.25 / 2.3 | Numerical and DataFrame processing |
    """)
