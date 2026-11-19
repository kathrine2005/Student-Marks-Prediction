# Student Marks Prediction Web App

A beginner-friendly Flask website that trains a machine-learning model from your CSV file and predicts student marks.

## Features
- Upload your own student CSV dataset.
- Select the target column (the marks column to predict).
- Select one or more input columns (study hours, attendance, previous marks, etc.).
- Choose regression (numeric marks) or classification (grade/category).
- Train a Random Forest model and view evaluation metrics.
- Predict from another CSV file or enter one student's details manually.
- View prediction results and download them as a CSV.
- Download a sample dataset to test the website.

## 1. Install Python
Use Python 3.10–3.13 (64-bit). Python 3.13 is supported by recent scientific Python packages. Avoid preview Python versions.

## 2. Open the project
Extract the ZIP file. Open the extracted `student_marks_prediction_webapp` folder in VS Code.

## 3. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -3.13 -m venv .venv
.venv\\Scripts\\Activate.ps1
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\\Scripts\\activate.bat
```

## 4. Install packages
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Run the website
```bash
python app.py
```
Open this address in your browser:
`http://127.0.0.1:5000`

## How to use
1. Click **Upload CSV** and choose your student dataset.
2. Select the column containing the marks you want to predict (for example, `Final_Marks`).
3. Select the input columns (for example, `Study_Hours`, `Attendance`, `Previous_Marks`).
4. Choose **Regression** for numeric marks and click **Train model**.
5. On the prediction page, either upload a new CSV containing the same input columns or enter one student's values manually.
6. Click **Predict**. View the results and click **Download predictions CSV**.

## CSV format example
Your training CSV should contain at least four usable rows, at least two columns, and a target column with values. For example:

| Study_Hours | Attendance | Previous_Marks | Final_Marks |
|---:|---:|---:|---:|
| 2 | 65 | 45 | 42 |
| 3 | 70 | 50 | 48 |
| 5 | 85 | 65 | 68 |

For prediction, the new CSV must contain the same feature/input column names selected during training. It does not need to contain the target column.

## Notes
- This is a local learning/demo application, not a production student assessment system.
- Model scores depend on dataset size and quality. A small sample dataset is only for testing the workflow.
- The app currently keeps the trained model and uploaded data in memory while the server is running. Restarting the app clears the current session.
