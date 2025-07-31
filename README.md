# 🤰 Collective-Pregnancy-Risk-Prediction using Deep Learning

This project aims to build predictive models using deep learning to assess health risks in pregnant women and their fetuses. It leverages two public datasets to train separate models for maternal and fetal health, and stacks them to generate a unified risk score.

## 📁 Datasets Used

- **Maternal Health Risk Dataset** – Contains health metrics of pregnant women like blood pressure, heart rate, and hypertension risk.
- **Fetal Health Dataset** – Includes fetal measurements such as heart rate, accelerations, and contractions.

## 📌 Key Steps

1. **Data Preprocessing**
   - Class balancing using resampling
   - Feature scaling with `StandardScaler`
   - Label encoding

2. **Model 1: Maternal Risk Classifier**
   - Trained using a DNN with ReLU and Dropout layers
   - Predicts risk level: `Low`, `Mid`, `High`

3. **Model 2: Fetal Health Classifier**
   - Predicts fetal condition mapped to three classes

4. **Fusion Mechanism**
   - Combines predictions from both models with weighted averaging
   - Converts into a risk score ranging from 0 (safe) to 1 (high risk)

5. **Model 3: Stacked Model**
   - Input: Combined features from maternal and fetal data
   - Output: Final risk score via a regression model

6. **Custom Prediction Function**
   - `stacking(fetal, maternal)` – Allows dynamic user input to get real-time prediction

## 📊 Model Evaluation

- Loss: Sparse categorical cross-entropy (Classification)
- Metrics: Accuracy for classification, MSE & MAE for regression

