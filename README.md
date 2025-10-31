# 🧠 DA5401 A7: Multi-Class Model Selection using ROC and Precision–Recall Curves

## 🏫 Course Information
**Course Code:** DA5401 
**Assignment:** A7 — Model Selection using ROC and Precision–Recall Curves  
**Dataset:** UCI Landsat Satellite Dataset (6 classes)  
**Student:** Ajaz  

---

## 🎯 Objective
This assignment focuses on performing **multi-class model selection** using the **Landsat Satellite dataset**.  
The primary goal is to compare several classifiers across **ROC (Receiver Operating Characteristic)** and **PRC (Precision–Recall Curve)** analyses to identify which models perform best, which perform poorly, and why.  
The study emphasizes **interpretation of performance curves**, not just accuracy metrics, to make informed decisions about model quality.

---

## 🗂️ Dataset Overview
- **Dataset:** Landsat Satellite (UCI ML Repository)  
- **Classes:** 6 distinct land-cover types (ignoring “all types present”)  
- **Features:** High-dimensional numerical features representing spectral measurements of satellite imagery  
- **Challenge:** Overlapping class boundaries and moderately imbalanced classes  

---

## 🧩 Models Compared
The following classifiers were trained and evaluated:

| No. | Model | Library | Expected Behavior |
|-----|--------|----------|-------------------|
| 1️⃣ | K-Nearest Neighbors (KNN) | `sklearn.neighbors` | Moderate to Good |
| 2️⃣ | Decision Tree | `sklearn.tree` | Moderate |
| 3️⃣ | Dummy Classifier | `sklearn.dummy` | Baseline / Random (AUC ≈ 0.5) |
| 4️⃣ | Logistic Regression | `sklearn.linear_model` | Good Linear Baseline |
| 5️⃣ | Gaussian Naive Bayes | `sklearn.naive_bayes` | Poor to Moderate |
| 6️⃣ | Support Vector Classifier (SVC) | `sklearn.svm` | Strong Nonlinear Classifier |

---

## ⚙️ Workflow Summary

### 🧱 Part A — Data Preparation and Baseline
1. **Data Loading:** Imported and standardized features from the Landsat dataset.  
2. **Train/Test Split:** Divided data into training and testing sets.  
3. **Model Training:** Trained all six classifiers with appropriate configurations.  
4. **Baseline Evaluation:** Calculated **Accuracy** and **Weighted F1-Score** to establish initial benchmarks.

---

### 📈 Part B — ROC Analysis for Model Selection
1. **One-vs-Rest (OvR) ROC Calculation:**  
   - Computed per-class ROC curves and macro-averaged AUC scores.  
2. **ROC Visualization:**  
   - Plotted all six models’ macro-averaged ROC curves together for comparison.  
3. **ROC Interpretation:**  
   - Identified the best model (**SVC**) with the highest macro-AUC (> 0.99).  
   - Dummy Classifier’s AUC (~0.5) represented random guessing.  
   - Explained the conceptual meaning of AUC < 0.5 — reversed decision boundary or systematic misclassification.

---

### 📊 Part C — Precision–Recall Curve (PRC) Analysis
1. **PRC Calculation:**  
   - Used `precision_recall_curve` and `average_precision_score` for multi-class (OvR) computation.  
   - Explained why PRC is more suitable for **imbalanced datasets** (focuses on positive class performance).  
2. **PRC Visualization:**  
   - Created a unified PRC plot comparing macro-averaged precision and recall across all models.  
3. **PRC Interpretation:**  
   - **KNN (AP ≈ 0.924)** and **SVC (AP ≈ 0.918)** achieved the highest average precision.  
   - **Logistic Regression** was stable but linear.  
   - **Decision Tree** and **GaussianNB** declined faster in precision.  
   - **Dummy Classifier** remained near the random baseline (AP ≈ 0.16).  
   - Highlighted how poor models show sharp precision drops as recall increases.

---

### 🧠 Part D — Final Recommendation
1. **Model Ranking Comparison:**  
   - Rankings from Accuracy, F1, ROC-AUC, and PRC-AP were mostly consistent.  
   - Minor differences appeared due to models performing differently across thresholds.  
2. **Best Model Recommendation:**  
   - **SVC** and **KNN** provided the best overall balance between sensitivity and specificity.  
   - **SVC** was ultimately chosen as the most robust model due to its superior generalization and smooth ROC/PRC curves.

---

### 🌟 Brownie Points Section
To go beyond the core task:
- Trained **RandomForest** and **XGBoost** models — both delivered near-perfect AUC (~0.995) and PRC-AP (~0.98).  
- Designed a **“Bad Model” (simple neural net)** with extremely poor configuration (1 hidden layer, 1 epoch, high learning rate) to produce AUC < 0.5, showcasing poor decision boundaries.

---

## 📉 Key Metric Comparison

| Model | Accuracy | Weighted F1 | ROC-AUC | PRC-AP |
|:------|:----------|:------------|:--------|:--------|
| **KNN** | 0.944 | 0.943 | 0.989 | 0.924 |
| **SVC** | 0.943 | 0.942 | 0.993 | 0.918 |
| **Logistic Regression** | 0.899 | 0.896 | 0.978 | 0.812 |
| **Decision Tree** | 0.890 | 0.889 | 0.900 | 0.737 |
| **GaussianNB** | 0.844 | 0.848 | 0.971 | 0.810 |
| **Dummy** | 0.301 | 0.139 | 0.500 | 0.167 |
| **RandomForest** ⭐ | 0.949 | 0.949 | 0.995 | 0.980 |
| **XGBoost** ⭐ | 0.942 | 0.942 | 0.995 | 0.980 |
| **Bad Model (NN)** ⚠️ | 0.194 | 0.199 | 0.490 | 0.200 |

---

## 🏁 Conclusion
This assignment demonstrated that:
- Evaluating models using **ROC** and **PRC curves** provides deeper insights than accuracy alone.  
- **SVC** and **KNN** emerged as the best-performing classifiers with high discriminative power.  
- **ROC-AUC** reflects overall model separability, while **PRC-AP** highlights class-specific precision–recall balance.  
- **Ensemble models (RF, XGBoost)** further enhanced performance, confirming their robustness.  
- Poor models (AUC < 0.5) illustrate inverted or ineffective learning behavior — useful for understanding model failure cases.

---



## 🏷️ Keywords
`Machine Learning` · `Model Selection` · `ROC-AUC` · `Precision-Recall` · `SVC` · `KNN` · `XGBoost` · `RandomForest`

---
