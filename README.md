
# Bati Bank - Credit Scoring Model

This project aims to develop a Credit Scoring Model to facilitate a buy-now-pay-later service in partnership with an upcoming eCommerce company. The model will help assess customer creditworthiness by utilizing historical transaction data and customer features.

## Project Overview

The main objective is to create a model that can categorize potential borrowers into high-risk (bad) and low-risk (good) categories. This project involves several steps, including data preprocessing, feature engineering, model training, and evaluation.

### Key Tasks
1. **Define a proxy variable** to categorize users as high risk or low risk.
2. **Select observable features** that serve as predictors for the defined default variable.
3. **Develop a model** to assign risk probabilities for new customers.
4. **Develop a credit scoring model** based on risk probability estimates.

## Installation

### Creating a Virtual Environment

#### Using Conda

If you prefer Conda as your package manager:

1. Open your terminal or command prompt.
2. Navigate to your project directory.
3. Run the following command to create a new Conda environment:

    ```bash
    conda create --name bati_bank_analysis python=3.12.5
    ```

4. Activate the environment:

    ```bash
    conda activate bati_bank_analysis
    ```

#### Using Virtualenv

If you prefer using `venv`, Python's built-in virtual environment module:

1. Open your terminal or command prompt.
2. Navigate to your project directory.
3. Run the following command to create a new virtual environment:

    ```bash
    python -m venv bati_bank_analysis
    ```

4. Activate the environment:

    - On Windows:
        ```bash
        .\bati_bank_analysis\Scripts\activate
        ```

    - On macOS/Linux:
        ```bash
        source bati_bank_analysis/bin/activate
        ```

### Installing Dependencies

Once your virtual environment is created and activated, install the required dependencies using:

```bash
pip install -r requirements.txt


## Project Structure
app/
├── models/
│   ├── Decision_Tree.pkl
│   ├── Logistic_Regression.pkl
│   ├── pca.pkl
│   ├── Random_Forest.pkl
│   └── scaler.pkl
├── __init__.py
├── app.py
├── rfm_calculation.py
├── requirements.txt
├── templates/
│   ├── input_form.html
│   └── result.html
notebooks/
├── EDA.ipynb
├── feature_engineering.ipynb
└── modeling.ipynb
scripts/
├── __init__.py
├── EDA.py
├── feature_engineering.py
└── modeling.py
├── README.md
├── .gitignore
├── Dockerfile
└── .dockerignore

