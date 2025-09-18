from data_prep import load_and_prepare_data
from models import get_models, train_and_evaluate
from visualize import plot_predictions

def main():
    # Load Data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Train & Evaluate Models
    models = get_models()
    for name, model in models.items():
        y_pred, trained_model = train_and_evaluate(name, model, X_train, y_train, X_test, y_test)

        # Only plot Random Forest (best model usually)
        if name == "Random Forest":
            plot_predictions(y_test, y_pred, name)

if __name__ == "__main__":
    main()
