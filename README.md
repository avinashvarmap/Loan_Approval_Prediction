# Loan_Approval_Prediction
📌 Objective:
To build a machine learning model that predicts whether a loan will be approved based on applicant details such as income, credit history, and employment status.

📊 Data Exploration & Preprocessing:
Initial Analysis: Visualized class distribution of Loan_Status, identifying class imbalance.

Missing Values: Imputed missing values using median (for numerical features) and mode (for categorical ones).

Categorical Encoding:

Binary encoding for features like Gender, Married, etc.

One-hot encoding for Property_Area.

Feature Engineering:

Created Total_Income = ApplicantIncome + CoapplicantIncome.

Applied log transformation on skewed features to normalize distributions.

Feature Selection: Dropped irrelevant columns like Loan_ID.

📈 Exploratory Data Analysis:
Used countplots, boxplots, and heatmaps to understand feature distributions and correlations.

Applied Chi-Square tests for categorical variables and T-tests for numerical ones to check significance against loan approval.

Found Credit_History, Total_Income, and LoanAmount to be strong predictors.

🧪 Model Building:
Data Splitting: Stratified train-test split (80/20) on Loan_Status.

Handling Imbalance: Applied SMOTE to balance the classes in the training set.

Feature Scaling: Standardized numerical features using StandardScaler.

🤖 Models Trained:
Logistic Regression (with pipeline + hyperparameter tuning)

Random Forest

XGBoost

Naive Bayes

Each model was:

Trained on SMOTE-balanced data

Evaluated on the test set using classification report and ROC-AUC score

🛠️ Model Tuning:
RandomizedSearchCV and GridSearchCV were used for hyperparameter tuning.

Each model’s parameters were optimized for ROC-AUC score using cross-validation.

✅ Final Model:
The best-performing model was a Logistic Regression classifier.

Feature importance was extracted from model coefficients and visualized.

The final model was saved using joblib for deployment:
loan_prediction_model.pkl
