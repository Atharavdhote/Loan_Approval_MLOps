
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/loan_approval.csv")

# Basic information
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicates:", df.duplicated().sum())

# Target distribution
print("\nLoan Status Distribution:")
print(df["Loan_Status"].value_counts())

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Loan Status Distribution
sns.countplot(data=df, x="Loan_Status")
plt.title("Loan Approval Distribution")
plt.savefig("outputs/loan_status.png")
plt.close()

# 2. Education vs Loan Status
sns.countplot(data=df, x="Education", hue="Loan_Status")
plt.title("Education vs Loan Approval")
plt.savefig("outputs/education_loan.png")
plt.close()

# 3. Credit History vs Loan Status
sns.countplot(data=df, x="Credit_History", hue="Loan_Status")
plt.title("Credit History vs Loan Approval")
plt.savefig("outputs/credit_history.png")
plt.close()

# 4. Applicant Income vs Loan Status
sns.boxplot(data=df, x="Loan_Status", y="ApplicantIncome")
plt.title("Applicant Income vs Loan Status")
plt.savefig("outputs/applicant_income.png")
plt.close()

# 5. Loan Amount vs Loan Status
sns.boxplot(data=df, x="Loan_Status", y="LoanAmount")
plt.title("Loan Amount vs Loan Status")
plt.savefig("outputs/loan_amount.png")
plt.close()

# 6. Property Area vs Loan Status
sns.countplot(data=df, x="Property_Area", hue="Loan_Status")
plt.title("Property Area vs Loan Status")
plt.savefig("outputs/property_area.png")
plt.close()

# 7. Correlation Heatmap
numeric = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm")
plt.title("Loan Dataset Correlation")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("\nEDA completed successfully!")
print("Charts saved in outputs folder.")
