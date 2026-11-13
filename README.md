# Loan Approval Prediction System – End-to-End MLOps

An end-to-end Machine Learning and MLOps project that predicts loan approval status using applicant information. The project includes Exploratory Data Analysis (EDA), data preprocessing, model training, REST API development, Docker containerization, automated testing, CI/CD, and cloud deployment.

## Live Application

- **Live API:** https://loan-approval-api-iz4x.onrender.com/
- **Swagger Documentation:** https://loan-approval-api-iz4x.onrender.com/docs
- **Health Check:** https://loan-approval-api-iz4x.onrender.com/health
- **GitHub Repository:** https://github.com/Atharavdhote/Loan_Approval_MLOps

## Project Overview

The system uses applicant details such as income, credit history, education, employment status, loan amount, and property area to predict loan approval status.

### Key Features

- Exploratory Data Analysis and visualization
- Missing-value treatment and categorical encoding
- Numerical feature scaling
- Machine Learning model comparison
- Logistic Regression, Decision Tree, and Random Forest
- FastAPI REST API with Pydantic validation
- Docker containerization
- Automated testing using Pytest
- GitHub Actions CI pipeline
- Cloud deployment using Render

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python 3.11 |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| API Development | FastAPI, Pydantic |
| Model Persistence | Joblib |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Version Control | Git, GitHub |

## Project Structure

```text
Loan_Approval_MLOps/
│
├── .github/
│   └── workflows/
│       └── main.yml
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── loan_approval.csv
│
├── models/
│   ├── loan_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── eda.py
│   ├── preprocess.py
│   └── train.py
│
├── outputs/
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Machine Learning Workflow

1. Load the loan prediction dataset.
2. Perform EDA and visualize important features.
3. Handle missing values using SimpleImputer.
4. Encode categorical features using OneHotEncoder.
5. Scale numerical features using StandardScaler.
6. Split the dataset into training and testing sets.
7. Train and compare three classification algorithms.
8. Select the best-performing model using F1 score.
9. Save the trained model and preprocessing pipeline.

## Model Evaluation

The project compares:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The model comparison results are saved in `outputs/model_comparison.csv`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict loan approval status |

## Run Locally Using Docker

### 1. Clone the repository

```bash
git clone https://github.com/Atharavdhote/Loan_Approval_MLOps.git
cd Loan_Approval_MLOps
```

### 2. Build Docker image

```bash
docker build -t loan-approval .
```

### 3. Run the API

```bash
docker run --rm -p 8000:8000 loan-approval
```

### 4. Open Swagger documentation

```text
http://localhost:8000/docs
```

## Automated Testing

Run the test suite inside Docker:

```bash
docker run --rm -v "${PWD}/models:/app/models" loan-approval python -m pytest tests/ -v
```

**Test result:** 4 tests passed.

## CI/CD Pipeline

GitHub Actions automatically:

1. Checks out the repository.
2. Builds the Docker image.
3. Runs the automated Pytest suite.

The workflow executes on pushes and pull requests targeting the `main` branch.

## Deployment

The application is deployed on Render using Docker.

Live URL: https://loan-approval-api-iz4x.onrender.com/

## Dataset

Loan Prediction Problem dataset from Kaggle:

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

## Disclaimer

This project is developed for educational and demonstration purposes. Its predictions should not be used as the sole basis for real-world lending or credit decisions. The dataset and model may contain biases and limitations.
