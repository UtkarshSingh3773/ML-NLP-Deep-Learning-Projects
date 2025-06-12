# House Price Prediction

This project demonstrates a machine learning pipeline for predicting house prices using a dataset with numerical and categorical features. The workflow includes data preprocessing, exploratory data analysis, feature engineering, and model training using several regression algorithms.

## Project Structure

- `House_Price_Prediction.ipynb`: Jupyter notebook containing all code, explanations, and visualizations for the pipeline.
- `HousePricePrediction.xlsx`: Source dataset used for training and evaluation (not included in this repo, please provide your own).

## Workflow Overview

### 1. Data Loading & Inspection
- **Libraries Used**: `pandas`, `matplotlib`, `seaborn`
- Loads the dataset from an Excel file and prints the first few rows to understand its structure.

### 2. Exploratory Data Analysis (EDA)
- Checks dataset dimensions and identifies variable types (categorical, integer, float).
- Visualizes correlations between numerical features using a heatmap.
- Analyzes unique values and distributions of categorical features via bar plots.

### 3. Data Cleaning & Preprocessing
- Drops the `Id` column as it is not useful for prediction.
- Handles missing values in the `SalePrice` column by filling with the mean.
- Removes rows with remaining missing values to create a clean dataset.
- Verifies that no missing values remain.

### 4. Feature Engineering
- Identifies and one-hot encodes categorical features using `sklearn.preprocessing.OneHotEncoder`.
- Final feature set consists of all numerical features and one-hot encoded categorical variables.

### 5. Model Training & Evaluation
- Splits the data into training and validation sets (80/20 split).
- Trains and evaluates three regression models:
  - **Support Vector Regressor (SVR)**
  - **Random Forest Regressor**
  - **Linear Regression**
- Evaluation metric: **Mean Absolute Percentage Error (MAPE)**

## Results

- Typical MAPE values for the models on the validation set:
  - **SVR**: ~0.19
  - **Random Forest**: ~0.19
  - **Linear Regression**: ~0.19

*These results may vary depending on the dataset and preprocessing steps.*

## How to Run

1. Place your dataset as `HousePricePrediction.xlsx` in the same directory as the notebook.
2. Open and run `House_Price_Prediction.ipynb` using Jupyter Notebook or Google Colab.
3. Follow the notebook cells for step-by-step execution and visualization.

## Requirements

- Python 3.7+
- pandas
- matplotlib
- seaborn
- scikit-learn

Install dependencies using:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Notes

- The notebook includes detailed comments and markdown explaining each step.
- You can further improve model performance by tuning hyperparameters or trying advanced feature engineering.

---

**Author:** utkarsh884  
**Repository:** [ML-NLP-Deep-Learning-Projects](https://github.com/utkarsh884/ML-NLP-Deep-Learning-Projects)
