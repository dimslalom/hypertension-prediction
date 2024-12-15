import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the dataset by handling missing values and scaling features."""
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Lembab']])
    
    # Replace original features with scaled features
    data['Lembab'] = scaled_features
    
    return data

def prepare_data(df):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Add seasonal features
    df['season'] = df['month'].apply(get_season)
    
    # Create lag features for humidity
    for i in range(1, 4):
        df[f'humidity_lag_{i}'] = df['Lembab'].shift(i)
    
    # Add rolling statistics
    df['humidity_rolling_mean'] = df['Lembab'].rolling(window=3).mean()
    df['humidity_rolling_std'] = df['Lembab'].rolling(window=3).std()
    
    # Drop rows with NaN from feature creation
    df = df.dropna()
    
    return df

def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

def split_data(data, target_column):
    """Split the dataset into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Example usage:
# data = load_data('data/train.csv')
# processed_data = preprocess_data(data)
# prepared_data = prepare_data(processed_data)
# X_train, X_test, y_train, y_test = split_data(prepared_data, 'target_column_name')