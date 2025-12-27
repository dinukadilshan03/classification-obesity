import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def calculate_bmi(X):
    """Calculates BMI from Height and Weight."""
    X = X.copy()
    # BMI formula: weight / height squared
    X['BMI'] = X['Weight'] / (X['Height'] ** 2)
    return X

# Create the transformer object
bmi_transformer = FunctionTransformer(calculate_bmi)

