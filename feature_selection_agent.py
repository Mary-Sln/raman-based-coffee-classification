import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_DIR = r"F:\AI-Agents\AI-Kaggle\Raman-IR-AI-Agent"
RES_DIR = os.path.join(BASE_DIR, "results")

def main():
    print("[+] Feature Selection Agent Initiated...")
    X_path = os.path.join(RES_DIR, "X_final_chemometrics.npy")
    y_path = os.path.join(RES_DIR, "y_labels.npy")
    
    if not os.path.exists(X_path):
        print("[-] Missing matrix. Please run preprocessing_agent.py first.")
        return
        
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"[+] Active Matrix Dimensions: {X.shape}")

    # Test different options for the optimal wavelength count
    k_options = [30, 50, 100, 150, 200, 300, 500]
    best_k = 785
    best_cv_score = 0
    
    print("\n[->] Screening Optimal Wavelength Count (K)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in k_options:
        fold_scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Feature selection strictly inside fold to prevent data leakage
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_sel)
            X_test_s = scaler.transform(X_test_sel)
            
            model = SVC(kernel='linear', random_state=42)
            model.fit(X_train_s, y_train)
            fold_scores.append(accuracy_score(y_test, model.predict(X_test_s)))
            
        mean_score = np.mean(fold_scores)
        print(f"    Tested Top {k} Wavelength Channels | Mean CV Accuracy: {mean_score:.4f}")
        
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_k = k

    print(f"\n[+] Optimization Complete! Champion Wavelength Count Selected: K = {best_k}")
    
    # Final fit and locking of selected feature matrix
    final_selector = SelectKBest(score_func=f_classif, k=best_k)
    X_selected = final_selector.fit_transform(X, y)
    
    np.save(os.path.join(RES_DIR, "X_selected_features.npy"), X_selected)
    print(f"[*] Saved Sub-Matrix ({X_selected.shape}) to results/X_selected_features.npy")

if __name__ == "__main__":
    main()