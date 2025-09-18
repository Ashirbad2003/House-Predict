import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    # Load dataset
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target, name="PRICE")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Performance:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")
    print("-" * 40)

    return y_pred

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = evaluate_model("Linear Regression", lr, X_test, y_test)

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = evaluate_model("Decision Tree", dt, X_test, y_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = evaluate_model("Random Forest", rf, X_test, y_test)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, color="blue", alpha=0.6, label="Random Forest Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("House Price Prediction (Random Forest)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
