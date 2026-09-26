# Loan Approval Prediction API

An end-to-end machine learning project that predicts whether a bank loan should be approved or rejected based on applicant data. 

This project demonstrates the complete AI engineering lifecycle: training a baseline Logistic Regression model, handling data scaling via pipelines, tuning probability thresholds for business optimization, and serving the model through a RESTful API.

## 🚀 Features
* **Data Preprocessing:** Uses Scikit-Learn's `ColumnTransformer` to apply `StandardScaler` to continuous variables (Income, Loan Amount) and `MinMaxScaler` to discrete numerical variables (Loan Term).
* **Threshold Tuning:** The model's decision threshold is manually optimized to find the perfect balance between catching bad loans (defaulters) and approving profitable customers.
* **FastAPI Backend (In Progress):** A web API that receives applicant JSON data, scales it on the fly, and returns a real-time prediction.

## 🛠️ Tech Stack
* **Machine Learning:** Python, Scikit-Learn, Pandas, NumPy
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Serialization:** Joblib

## 📊 Model Performance
The final optimized model achieves the following metrics on the test dataset:
* **Accuracy:** 84.3%
* **Precision (Class 1):** 0.89
* **Recall (Class 1):** 0.95
* **Strategic Trade-off:** The threshold was tuned to maximize True Negatives (catching bad loans) while minimizing False Negatives (incorrectly rejecting good customers).

## 📂 Project Structure
```text
logistic-regression-project/
│
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and model training
├── models/                 # Saved .pkl files (model and preprocessor)
├── main.py                 # FastAPI application setup
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 💻 Local Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/wahabali786110/logistic-regression-project.git
cd logistic-regression-project
```

**2. Create a virtual environment & install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the API server:**
```bash
uvicorn main:app --reload
```
*Navigate to `http://127.0.0.1:8000/docs` to test the API endpoints.*