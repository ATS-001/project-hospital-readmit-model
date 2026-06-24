# project-hospital-readmit-model 
# *aka* Project Pulse: Readmission Risk Console

An end-to-end **Machine Learning web application** designed as a research-prototype decision support tool to assist clinical discharge planners in assessing **30-day hospital readmission risk**. By leveraging a trained **XGBoost classifier**, the system processes encounter-specific data to provide probabilistic insights directly at the point of care.

---

## ⚠️ Important Clinical Disclaimer
This tool is for **decision support only** and does not replace professional clinical judgment.  
Results are probabilistic, derived from historical encounter data, and should be used to support—not dictate—discharge planning and chart review protocols.  
This application is a **research prototype** and is not currently intended for clinical deployment. Please refer to official hospital protocols for final discharge decisions.

---

## 📋 Features
- **Demographics & Age:** Tracks age groups, gender, and race.  
- **Admission & Discharge Details:** Includes admission type, source, and discharge disposition.  
- **Utilization Metrics:** Captures historical encounter data such as lab procedures, previous outpatient/inpatient visits, emergency visits, and total number of diagnoses.  
- **Clinical Data:** Houses primary/secondary/additional diagnosis categories and an extensive list of diabetes-related medications.  
- **Risk Assessment Output:** Generates a 30-day readmission probability percentage with confidence levels, highlighting borderline results for further review.  

---

## 📊 How It Works
1. **Input:** Discharge planners enter relevant encounter data into the console.  
2. **Assessment:** The underlying XGBoost model evaluates the input against historical patterns.  
3. **Output:** The system generates a readmission probability percentage and confidence level.  

---

## 🖥️ Website
- **Deployed using:** Streamlit
- **Link:** https://project-hospital-readmit-model-ats001.streamlit.app/

## 🛠 Technology Stack
- **Framework:** Streamlit  
- **Model:** XGBoost  
- **Dataset:** Diabetes 130-US Hospitals (1999–2008)  
- **Type:** Research Prototype (Not for clinical deployment)  

---

## 📅 Workshop Context
**Day 2 of Projectathon conducted by μLearn LBSITW, AI x DS (24th June 2026)**  
- **Presented by:** *Aiswarya Jayaprakash, Data Science IG Lead μLearn LBSITW*  
- **Focus:** Transitioning from image data to structured clinical records to build a supervised learning classification model.  

### Core Concepts Mastered
- **Supervised Learning:** Training models on historical, labeled data.  
- **Regression vs. Classification:** Differentiating continuous numeric targets from binary outcomes (predicting "Yes/No" readmission).  
- **Unsupervised Learning:** Identifying hidden structures in unlabeled data.  
- **Reinforcement Learning:** Training agents via trial-and-error rewards/penalties.  

---

## 🛠️ Machine Learning Pipeline Architecture
1. **Feature Engineering & Data Cleaning**  
   - Parsed age ranges (e.g., `[10-20)`) into numerical means.  
   - Remapped ICD-9 diagnosis codes into 8 diagnostic groups (*Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms*).  

2. **Categorical Encoding**  
   - Converted medication flags (`No`, `Steady`, `Up`, `Down`) into ordinal features (`0`, `1`, `2`, `3`).  
   - Applied one-hot encoding to nominal text variables with `drop_first=True`.  

3. **Feature Selection & Imbalance Mitigation**  
   - Removed highly collinear features (>0.85 correlation).  
   - Balanced skewed `readmitted` labels using `scale_pos_weight`.  

4. **Model Training & Evaluation**  
   - 80/20 train-test split.  
   - Standardized features with `StandardScaler`.  
   - Trained `XGBClassifier` (`max_depth=5`, `n_estimators=200`, `learning_rate=0.1`).  
   - Evaluated with Accuracy, Precision, Recall, and F1-Score.  

5. **Artifact Serialization**  
   - Exported trained model, scaler, and feature schema into `.pkl` binaries via `joblib`.  

---

## 📁 Repository Directory Structure
```text
├── diabetes_app/
│   ├── app.py                  # Streamlit application entry point
│   ├── model.pkl               # Serialized XGBoost model binary
│   ├── scaler.pkl              # Fitted StandardScaler object 
│   └── feature_columns.pkl     # Exported pipeline feature matrix schema
├── notebooks/
│   └── pipeline.ipynb          # Notebook documenting EDA, cleaning, and training
├── requirements.txt            # Python environment dependencies
└── README.md                   # Repository documentation

---

## 🚀 Local Installation & Deployment

1. **Clone this repository:**
   ```bash
      git clone https://github.com/ATS-001/project-hospital-readmit-model.git
      cd project-hospital-readmit-model
   ```

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Streamlit Cloud Deployment

* **Repository:** `project-hospital-readmit-model`
* **Branch:** `main`
* **Main file path:** `app.py`

---

## 🔮 Acknowledgments

Special thanks to Aishwarya Jayaprakash (https://github.com/Aiswarya-Jayaprakash) for the baseline pipeline code, structural guidelines, and workshop instruction that enabled this end-to-end model and deployment.

---

## 👤 Author
**Aaron Thalakkottor Sooraj" - B.Tech Computer Science & Engineering Student

---
