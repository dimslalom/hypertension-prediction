import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np

# Load the training data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Train the model
def train_model(data):
    X = data[['Lembab']]  # Features
    y = data['target']  # Target variable (to be defined)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f'R² Score: {r2}')

    return model

def train_models(X_train, y_train):
    models = {
        'rf': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        # Perform cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
        
        # Fit model on full training data
        model.fit(X_train, y_train)
        results[name]['model'] = model
        
    return results

# Save the model
def save_model(model, file_path):
    joblib.dump(model, file_path)

if __name__ == "__main__":
    # Define file paths
    train_file_path = '../data/train.csv'
    model_file_path = '../models/hypertension_model.pkl'

    # Load data
    data = load_data(train_file_path)

    # Train the model
    model = train_model(data)

    # Save the model
    save_model(model, model_file_path)