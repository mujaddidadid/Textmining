import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone

def train_and_compare_models(X, y, test_size=0.2, random_state=42):
    """
    Train multiple models and compare accuracy with Smart Tie-Breaking.
    Jika akurasi Stacking == Decision Tree, Stacking akan dipilih karena lebih robust.
    """
    
    # 1. VALIDASI DATA
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y

    if len(X) < 10:
        raise ValueError(" Data terlalu sedikit (minimal 10 sampel).")

    if (X < 0).any():
        print(" Warning: Data mengandung nilai negatif. MultinomialNB mungkin tidak akurat.")

    unique, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)
    
    # 2. SPLIT DATA
    stratify_param = y if min_samples >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # 3. DEFINISI BASE MODELS
    base_models_def = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
    }

    results = {}
    trained_models = {}

    # 4. TRAIN BASE MODELS
    for name, model in base_models_def.items():
        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = acc
        trained_models[name] = clf

    # 5. TRAIN STACKING (OPTIMIZED)
    estimators_list = [
        ('nb', clone(base_models_def["Naive Bayes"])),
        ('dt', clone(base_models_def["Decision Tree"])),
        ('rf', clone(base_models_def["Random Forest"]))
    ]

    n_folds = 5
    if len(X_train) < 50: n_folds = 3
    if len(X_train) < 20: n_folds = 2
    
    cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    stacking_model = StackingClassifier(
        estimators=estimators_list,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv_strategy,
        n_jobs=-1
    )

    try:
        stacking_model.fit(X_train, y_train)
        y_pred_stack = stacking_model.predict(X_test)
        acc_stack = accuracy_score(y_test, y_pred_stack)
        
        results["Stacking"] = acc_stack
        trained_models["Stacking"] = stacking_model
    except Exception as e:
        print(f"Stacking error: {e}")
        results["Stacking"] = 0.0

    # 6. PEMILIHAN BEST MODEL 
    
    # Cari skor tertinggi
    max_acc = max(results.values())
    
    top_models = [name for name, acc in results.items() if acc == max_acc]
    
    if "Stacking" in top_models:
        best_model_name = "Stacking"
    elif "Random Forest" in top_models:
        best_model_name = "Random Forest"
    else:
        best_model_name = top_models[0] 

    best_model_instance = trained_models[best_model_name]
    best_acc = results[best_model_name]

    # 7. OUTPUT
    y_pred_best = best_model_instance.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best)

    accuracy_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": list(results.values())
    }).sort_values(by="Accuracy", ascending=False)

    print(f"🏆 Pemenang: {best_model_name} dengan Akurasi: {best_acc:.4f}")

    return {
        "accuracy_df": accuracy_df,
        "best_model": best_model_name,
        "best_accuracy": best_acc,
        "confusion_matrix": cm,
        "classification_report": report
    }