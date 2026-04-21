from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")

    # Simple condition for CI fail/pass
    assert acc > 0.5, "Model accuracy too low!"

if __name__ == "__main__":
    main()
