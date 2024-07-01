

# BigMart Sales Prediction

This project aims to build a machine learning model to predict the sales of products in BigMart outlets. By leveraging various machine learning algorithms and data preprocessing techniques, the project provides an effective tool for forecasting sales based on historical data.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The BigMart Sales Predictor uses machine learning techniques to analyze historical sales data and predict future sales. This tool is designed to assist business analysts, data scientists, and retail managers in making data-driven decisions to optimize inventory and sales strategies.

## Features
### Data Preprocessing:
- Handling missing values
- Encoding categorical variables
- Feature scaling

### Model Training:
- Splitting the dataset into training and testing sets
- Training various machine learning models (e.g., Linear Regression, Decision Trees, Random Forest)
- Evaluating model performance using metrics like RMSE, MAE, and R²

### Prediction System:
- Input features for sales prediction
- Predicting the sales of products

## Installation
To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Vdntrai/BigMart_SalesPred.git
    cd BigMart_SalesPred
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To use the BigMart Sales Predictor, follow these steps:

1. Run the Jupyter Notebook:
    ```bash
    jupyter notebook BigMart_SalesPred.ipynb
    ```

2. Import relevant dependencies:
    ```python
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ```

3. Loading the data:
    ```python
    data = pd.read_csv('path_to_data.csv')
    ```

4. Train the model:
    ```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

5. Evaluate model performance:
    ```python
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}')
    ```

6. Make predictions:
    ```python
    new_data = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
    sales_pred = model.predict(new_data)
    print(f'Predicted Sales: {sales_pred[0]:.2f}')
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
