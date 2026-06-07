# DENV-PRED 1.0 (Dengue, Inhibitor Predictor for Pan-Serotype NS3/NS5, version 1.0)

### Overview
DENV-PRED 1.0 is a dual-architecture structure-based machine learning (SBML) framework developed for the high-stringency identification of pan-serotype inhibitors targeting Dengue virus (DENV) NS3 protease and NS5 polymerase. By integrating 2D cheminformatics with 3D stereochemical pocket constraints, the pipeline acts as a thermodynamic gatekeeper, prioritizing nanomolar-affinity leads while aggressively filtering out generic scaffolds and structural decoys.

### Features
* **Pan-Serotype Coverage:** Predicts inhibitory potential across all four DENV serotypes (DENV 1-4).
* **SBML Engine:** Maps ligands against explicit 3D viral pocket constraints (SASA volume, H-bond donor/acceptor arrays, and aromatic networks).
* **Dual-Architecture Inference:** * **Gatekeeper:** A Gradient Boosting Classifier optimized for F1-Score to minimize false positives.
    * **Ranker:** A Gradient Boosting Regressor for continuous $pIC_{50}$ binding affinity estimation.
* **Rigorous Validation:** Incorporates an Isolation Forest-based Applicability Domain (AD) to prevent false positives from out-of-distribution chemical structures.
* **Transparent SAR:** Integrated SHAP (SHapley Additive exPlanations) analysis for structural interpretation of inhibitor activity.

---

### Access the Web Tool
You can access the DENV-PRED 1.0 virtual screening pipeline directly through your web browser:

🔗 **[Launch DENV-PRED 1.0 Web Tool Here](https://denv-predict-1-sbml-prediction-model.streamlit.app/])**

---

### Citation
If you utilize the DENV-PRED 1.0 webtool in your research, please cite:

> **DENV-PRED 1.0 Webtool** | Ajwin Joseph Martin, Dileep Kumar | Manipal Academy of Higher Education (MAHE) | Version 1.0 (2026).  
> **Webtool URL:** *(https://[YOUR-STREAMLIT-URL-HERE])*


---

### Copyright & Intellectual Property

**© 2026 Manipal Academy of Higher Education (MAHE). All rights reserved.**

**Authors/Creators:** Ajwin Joseph Martin and Dileep Kumar

The source code, algorithms, consensus logic, and trained models associated with DENV-PRED 1.0 are the exclusive intellectual property of Manipal Academy of Higher Education (MAHE). 

**Permissions:**
* You are permitted to view the source code for educational and peer-review purposes.
* You are permitted to use the deployed web tool via the provided URL for virtual screening tasks, provided proper citation is given.

**Restrictions:**
* You may **NOT** copy, reproduce, distribute, modify, or create derivative works from this codebase.
* You may **NOT** use the code or models for any commercial or private non-commercial deployment without explicit written permission from the copyright owner (MAHE) and the authors.

For licensing inquiries or permission requests, please contact the authors directly.
