import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label=f"{model_name} Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"House Price Prediction ({model_name})")
    plt.legend()
    plt.show()
