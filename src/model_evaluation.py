from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    results = {
        'r2_score': r2,
        'rmse': rmse,
        'predictions': predictions
    }
    
    plot_predictions(y_test, predictions)
    
    return results

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()

def load_test_data(file_path):
    """
    Load the test dataset.

    Parameters:
    file_path (str): Path to the test dataset CSV file.

    Returns:
    DataFrame: Loaded test dataset.
    """
    return pd.read_csv(file_path)

def main():
    # Example usage
    test_data = load_test_data('../data/test.csv')
    # Assuming y_true and y_pred are defined
    # r2 = evaluate_model(y_true, y_pred)
    # print(f'R² Score: {r2}')

if __name__ == "__main__":
    main()