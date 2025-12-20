import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_and_compare_models(X, y, test_size=0.2, random_state=42):
    """
    Train multiple models and compare accuracy
    """

    # ==============================
    # SPLIT DATA
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ==============================
    # MODEL DEFINITIONS
    # ==============================
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    results = {}
    trained_models = {}

    # ==============================
    # TRAIN BASE MODELS
    # ==============================
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results[name] = acc
        trained_models[name] = model

    # ==============================
    # STACKING CLASSIFIER
    # ==============================
    estimators = [
        ("nb", models["Naive Bayes"]),
        ("dt", models["Decision Tree"]),
        ("rf", models["Random Forest"]),
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )

    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)

    results["Stacking"] = acc_stack
    trained_models["Stacking"] = stacking

    # ==============================
    # BEST MODEL
    # ==============================
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    # ==============================
    # FINAL EVALUATION
    # ==============================
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best)

    accuracy_df = pd.DataFrame({
        "Model": results.keys(),
        "Accuracy": results.values()
    }).sort_values(by="Accuracy", ascending=False)

    return {
        "accuracy_df": accuracy_df,
        "best_model": best_model_name,
        "best_accuracy": results[best_model_name],
        "confusion_matrix": cm,
        "classification_report": report
    }
