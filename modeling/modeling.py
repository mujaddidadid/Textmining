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
    Train multiple models and compare accuracy with Optimized Stacking.
    """
    
    # ==============================
    # 1. VALIDASI DATA ROBUST
    # ==============================
    print(f"=== VALIDASI DATA ===")
    
    # Konversi ke numpy array
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y

    print(f"Dimensi Data: X={X.shape}, y={y.shape}")

    # Validasi Dasar
    if len(X) < 10: 
        raise ValueError(f"❌ Data terlalu sedikit! Minimal 10 sampel untuk Stacking yang efektif.")
    if len(X) != len(y):
        raise ValueError(f"❌ Jumlah X ({len(X)}) dan y ({len(y)}) tidak sama!")

    # Cek Distribusi Kelas
    unique_labels, label_counts = np.unique(y, return_counts=True)
    min_samples_per_class = np.min(label_counts)
    print(f"Kelas: {unique_labels}, Counts: {label_counts}")

    if (X < 0).any():
        print("⚠️ PERINGATAN: Data mengandung nilai negatif. MultinomialNB mungkin error.")
    
    # ==============================
    # 2. SPLIT DATA
    # ==============================
    print(f"=== SPLIT DATA ===")
    
    can_use_stratify = min_samples_per_class >= 2
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_use_stratify else None
        )
        print(f"Split OK: Train={len(X_train)}, Test={len(X_test)}")
    except Exception as e:
        print(f"❌ Error Split: {e}")
        return None

    # ==============================
    # 3. DEFINISI BASE MODELS
    # ==============================
    base_models_def = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
    }

    results = {}
    trained_models = {}

    # ==============================
    # 4. TRAIN INDIVIDUAL MODELS
    # ==============================
    print("=== TRAINING BASE MODELS ===")
    for name, model in base_models_def.items():
        clf = clone(model) 
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = acc
        trained_models[name] = clf
        print(f"✅ {name}: {acc:.4f}")

    # ==============================
    # 5. STACKING CLASSIFIER (OPTIMIZED)
    # ==============================
    print("=== TRAINING STACKING CLASSIFIER ===")
    
    estimators_list = [
        ('nb', clone(base_models_def["Naive Bayes"])),
        ('dt', clone(base_models_def["Decision Tree"])),
        ('rf', clone(base_models_def["Random Forest"]))
    ]

    n_folds = 5
    if len(X_train) < 50:
        n_folds = 3
    if len(X_train) < 20:
        n_folds = 2
    
    cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    stacking_model = StackingClassifier(
        estimators=estimators_list,
        final_estimator=LogisticRegression(), 
        cv=cv_strategy,    
        stack_method='auto', 
        n_jobs=-1          
    )

    try:
        stacking_model.fit(X_train, y_train)
        y_pred_stack = stacking_model.predict(X_test)
        acc_stack = accuracy_score(y_test, y_pred_stack)
        
        results["Stacking"] = acc_stack
        trained_models["Stacking"] = stacking_model
        print(f"✅ Stacking Classifier: {acc_stack:.4f}")
        
    except Exception as e:
        print(f"❌ Stacking Failed: {e}")
        results["Stacking"] = 0.0

    # ==============================
    # 6. EVALUASI AKHIR
    # ==============================
    # Mencari nama model terbaik
    best_model_name = max(results, key=results.get)
    best_model_instance = trained_models.get(best_model_name)
    best_acc = results[best_model_name]

    print(f"\n🏆 BEST MODEL: {best_model_name} ({best_acc:.4f})")

    y_pred_best = best_model_instance.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best, output_dict=True)

    accuracy_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": list(results.values())
    }).sort_values(by="Accuracy", ascending=False)

    return {
        "accuracy_df": accuracy_df,
        "best_model": best_model_name,      # <--- DIPERBAIKI (Sebelumnya 'best_model' tanpa '_name')
        "best_model_instance": best_model_instance, 
        "best_accuracy": best_acc,
        "confusion_matrix": cm,
        "classification_report": report
    }