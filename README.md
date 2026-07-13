# Heart Disease Prediction Web App

This project is a Streamlit web application for predicting heart disease using machine learning. It includes:

- A prediction interface for new patients
- Exploratory data analysis visualizations
- Comparison of three models: Logistic Regression, Random Forest, and XGBoost
- Model performance metrics and evaluation details

## Dataset

The app uses the file `heart_cleaned.csv`.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py` - Main Streamlit application
- `heart_cleaned.csv` - Dataset used for training and evaluation
- `requirements.txt` - Python dependencies
- `.gitignore` - Files to ignore in Git

## Notes

The app uses an 80/20 train-test split for model evaluation.
