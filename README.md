
# 💳 Credit Card Fraud Detection

A data science project that leverages machine learning algorithms to detect fraudulent credit card transactions from a highly imbalanced dataset.

![fraud](https://img.shields.io/badge/Fraud-Detection-red) ![ML](https://img.shields.io/badge/Machine-Learning-blue) ![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)

## 📌 Project Overview

This project tackles the challenge of identifying fraudulent transactions in a credit card dataset using various machine learning techniques. Given the highly imbalanced nature of the data (fraudulent transactions make up a tiny fraction), the project focuses on strategies to enhance classification performance and minimize false negatives.

---

## 🗃️ Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Frauds**: 492 (0.172%)
- **Features**: 
  - 28 anonymized numerical variables (V1 to V28 from PCA transformation)
  - `Time` and `Amount`
  - `Class`: Target variable (0 = legitimate, 1 = fraud)

---

## ⚙️ Tools & Libraries Used

- Python 3.x
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## 🧪 Key Steps & Workflow

1. **Exploratory Data Analysis (EDA)**
   - Class distribution visualization
   - Correlation heatmap
   - Boxplots for outlier detection

2. **Preprocessing**
   - Feature scaling (`Amount` with StandardScaler)
   - Data splitting (train-test split)

3. **Modeling**
   - **Random Forest Classifier**
   - **XGBoost Classifier**

4. **Evaluation**
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC-AUC Score

---

## 📈 Model Performance

| Model             | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|------------------|----------|-----------|--------|----------|---------|
| Random Forest     | ~99.97%  | 87.10%    | 89.47% | 88.27%   | 94.74%  |
| XGBoost           | ~99.96%  | 84.00%    | 86.84% | 85.40%   | 93.42%  |

*Note: Metrics may vary slightly depending on random state and hyperparameters.*

---

## 📊 Visualizations

- **Class Distribution Plot**
- **Feature Correlation Heatmap**
- **Boxplot for PCA Features**
- **ROC Curves for Models**
- **Confusion Matrix Heatmaps**

---

## 💡 Insights

- Class imbalance is significant — only 0.172% of the dataset are frauds.
- Ensemble models (Random Forest, XGBoost) offer high precision and recall.
- Proper resampling techniques or anomaly detection models could be further explored to improve recall.

---

## 📁 Project Structure

```
📦 Credit-Card-Fraud-Detection
 ┣ 📜 Credit_Card_Fraud_Detection.ipynb
 ┣ 📜 README.md
```

---

## 🚀 Getting Started

### ✅ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### ▶️ Run

```bash
git clone https://github.com/ElSabahNoor15/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

---

## 📌 Future Work

- Implement SMOTE or ADASYN for oversampling
- Try deep learning models (e.g., autoencoders, LSTM)
- Real-time fraud detection simulation
- Hyperparameter tuning with GridSearchCV

---

## 🙌 Acknowledgements

- Dataset provided by **UCI Machine Learning Repository** and **Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles)**.

---

## 📬 Contact

Feel free to connect via [LinkedIn](https://www.linkedin.com/) or raise an issue for suggestions!
