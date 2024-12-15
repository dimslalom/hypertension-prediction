# Hypertension Prediction Project

This project aims to predict the prevalence of hypertension based on historical humidity data. The analysis leverages machine learning techniques to understand the relationship between humidity levels and hypertension rates, contributing to public health strategies.

## Project Structure

- **data/**: Contains the datasets used for training and testing the model.
  - **train.csv**: Training dataset with historical humidity data.
  - **test.csv**: Testing dataset for making predictions.
  - **sample_submission.csv**: Template for submission format.

- **notebooks/**: Jupyter notebooks for exploratory data analysis.
  - **analysis.ipynb**: Notebook for visualizing trends and patterns in the dataset.

- **src/**: Source code for data processing, model training, and evaluation.
  - **data_preprocessing.py**: Script for cleaning and preparing the dataset.
  - **model_training.py**: Script for training the predictive model.
  - **model_evaluation.py**: Script for evaluating model performance.
  - **utils.py**: Utility functions for various tasks.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd hypertension-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

- Use the `data_preprocessing.py` script to prepare the data before training the model.
- Train the model using `model_training.py`.
- Evaluate the model's performance with `model_evaluation.py`.
- Explore the dataset and visualize results in `analysis.ipynb`.

## Contribution

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.