# Heart-Disease-Prediction-using-Logistic-Regression
Heart Disease Prediction using Logistic Regression

Developed by: Arvind Sharma
Tools Used: Python, Scikit-learn, Streamlit, Pandas, NumPy, Matplotlib, Seaborn
Dataset: Framingham Heart Study Dataset

Project Overview

Heart disease remains one of the leading causes of death globally. This project uses Machine Learning — specifically Logistic Regression — to predict the 10-year risk of Coronary Heart Disease (CHD) based on patient health metrics.

The goal is to provide a simple yet effective predictive model to assist clinicians and patients in early detection and prevention of heart disease.

 Objective

Develop a Logistic Regression model to predict whether a patient is at risk of developing CHD within the next 10 years using health-related features from the Framingham Heart Study dataset.

Project Workflow
1. Importing Libraries and Dataset

Libraries Used:

Pandas – Data manipulation

NumPy – Numerical operations

Matplotlib / Seaborn – Visualization

Scikit-learn – Model building and evaluation

Dataset: Framingham Heart Study
Records: 4,240 samples
Features: Age, Gender, Cholesterol, Blood Pressure, BMI, Smoking, Glucose
Target Variable: TenYearCHD (1 = At Risk, 0 = No Risk)

2. Data Preprocessing

Dropped irrelevant column (education)

Handled missing values (NaN removal)

Normalized numerical columns using StandardScaler

Encoded categorical variables

Split dataset: 70% training, 30% testing

3.  Exploratory Data Analysis (EDA)

EDA revealed that the likelihood of CHD increases with:

Higher age and cholesterol levels

Smoking habits

High blood pressure and glucose

Visualization Techniques:

Histograms and Count Plots

Correlation Heatmaps

Pairplots for risk factor analysis

4. Model Training – Logistic Regression

Algorithm: Logistic Regression

Loss Function: Binary Cross-Entropy (Log Loss)

Optimization: Gradient Descent (LogisticRegression() from sklearn)

Output: Probability between 0 and 1 using the Sigmoid Function

5.  Model Evaluation
Metric	Score
Accuracy	85.4%
Precision	83.1%
Recall	81.6%
F1-Score	82.3%
ROC-AUC	0.89

Visual Evaluations:

Confusion Matrix

ROC-AUC Curve

Feature Importance Graph

The model effectively distinguishes between high-risk and low-risk patients.

6. 🔍 Model Comparison
Model	Accuracy
Decision Tree	80%
Random Forest	84%
Logistic Regression	85% 

 Reason for Choosing Logistic Regression:

Simple and interpretable

Clinically explainable coefficients

Balanced performance without overfitting

7.  Streamlit Web App Deployment

An interactive Streamlit web app was built for real-time prediction.

App Features:

User-friendly form interface

Real-time prediction with confidence score

Visual risk alert:  High Risk |  Low Risk

Uses saved model files: model.pkl and scaler.pkl

8.  Prediction Example

Input:

Age = 55
Male = 1
Smoker = 1
Cholesterol = 250
BP = 145/90
BMI = 28.5
Glucose = 120


Predicted Output:

“High Risk of Heart Disease (Probability = 78%)”

This output can assist doctors in recommending early lifestyle and treatment interventions.

Visual Insights

ROC-AUC Curve: Model’s class separation ability

Confusion Matrix: Classification accuracy

Feature Importance: Identified key predictors

Model Comparison Plot: Performance visualization

 Future Enhancements

Implement Ensemble Models (Random Forest, XGBoost)

Integrate IoT-based health sensors for real-time monitoring

Develop a mobile app for patients and clinicians

Experiment with Deep Learning models (ANN, CNN)

 Repository Structure
 Heart-Disease-Prediction/
│
├──  Heart_disease_final.ipynb      # Jupyter notebook with full implementation
├──  Heart_Disease_Prediction_Presentation_Explained.pptx
├──  Heart_Disease_Prediction_Report_Arvind_Sharma.pdf
├── data/
│   └── framingham.csv               # Dataset
├──  model/
│   ├── model.pkl
│   └── scaler.pkl
├──  app.py                         # Streamlit web app
├──  requirements.txt               # Dependencies
└──  README.md                      # Project documentation

Installation & Usage

Clone the repository:

git clone https://github.com/yourusername/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction


Install dependencies:

pip install -r requirements.txt


Run the Streamlit App:

streamlit run app.py

 Conclusion

This project demonstrates how machine learning can contribute to preventive healthcare.
The Logistic Regression model (85% accuracy) provides interpretable and actionable predictions, making it suitable for clinical decision support.



https://github.com/user-attachments/assets/999a2ba0-67e7-4842-b7ab-8d55047dac6e

Training video is in PPT
