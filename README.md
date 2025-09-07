📊 Credit Card Fraud Risk Analysis (ML + BI)
🔎 Overview

This project combines Machine Learning and Business Intelligence (BI) to analyze and detect credit card fraud risk.
It provides:

1. A Power BI dashboard for interactive fraud insights.
2. A Streamlit ML web app for fraud detection using machine learning models.

📌 Features
🔹 Power BI Dashboard

1. Fraud risk analysis by fraud type, state, and month

2. Fraudulent transaction insights with critical/high/medium/low segmentation

3. Drill-down filters for fraud type, merchant, and state

🔹 Streamlit ML App

1. Upload CSV or use demo dataset

2. Train models (e.g., Random Forest, XGBoost, Logistic Regression)

3. Performance metrics: Precision, Recall, ROC AUC

4. ROC Curve visualization

5.Adjustable test split & decision threshold

🛠️ Tech Stack

1. Python (Pandas, Scikit-learn, Streamlit)

2. Power BI

3. Machine Learning Models (Random Forest, Logistic Regression, etc.)
   
🚀 How to Run
1️⃣ Power BI Dashboard

*Open Credit Card Fraud Risk Analysis.pbix in Power BI Desktop

*Explore fraud risk insights

2️⃣ Streamlit ML App
# Clone the repo
git clone https://github.com/your-username/Fraud-risk-analysis-ML-BI.git
cd Fraud-risk-analysis-ML-BI

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📸 Screenshots
🔹 Power BI Dashboard   ![Fraud analysis](https://github.com/Sanjana-Sen/Fraud-risk-analysis-ML-BI/blob/main/Risky.png)

🔹 Streamlit ML App  ![Detection](https://github.com/Sanjana-Sen/Fraud-risk-analysis-ML-BI/blob/main/Detect.png)

📈 Results

Fraud detection achieved Precision = 1.0, Recall = 1.0, ROC AUC = 1.0 on test dataset.

Identified Card Not Present as the top fraud type.

🔮 Future Improvements

Deploy Streamlit app on Streamlit Cloud/Heroku

Integrate real-time fraud detection API

Expand Power BI dashboard with geospatial fraud analysis

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.
