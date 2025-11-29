from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print(f"\n{name}")
        print(classification_report(y_test, preds))

        results.append({"Model": name, "Accuracy": acc})

    return pd.DataFrame(results)
