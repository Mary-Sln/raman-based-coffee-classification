import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Core path configurations matching your local setup
BASE_DIR = r"F:\AI-Agents\AI-Kaggle\Raman-IR-AI-Agent"
RES_DIR = os.path.join(BASE_DIR, "results")

def apply_snv(X):
    """Standard Normal Variate (SNV) to correct scatter effects"""
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def apply_msc(X):
    """Multiplicative Scatter Correction (MSC) against mean reference"""
    ref_spectrum = np.mean(X, axis=0)
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        fit = np.polyfit(ref_spectrum, X[i], 1, full=True)
        X_msc[i] = (X[i] - fit[0][1]) / fit[0][0]
    return X_msc

def evaluate_pipeline(X, y_labels):
    """Rigorous evaluation loop using Stratified 5-Fold Cross Validation"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, sens, specs, precs = [], [], [], [], []
    
    for train_idx, test_idx in skf.split(X, y_labels):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        # Scale strictly within the fold to avoid data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1500, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        sens.append(recall_score(y_test, y_pred, average='macro'))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        
        # Calculate Multi-Class Specificity
        cm = confusion_matrix(y_test, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)  
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specs.append(np.mean(tn / (tn + fp + 1e-8)))

    return np.mean(accs), np.mean(f1s), np.mean(sens), np.mean(specs), np.mean(precs)

def main():
    print("[+] Ingesting baseline-corrected array from previous stage...")
    X_path = os.path.join(RES_DIR, "X_preprocessed.npy")
    y_path = os.path.join(RES_DIR, "y_labels.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("[-] Error: Run 'skills/run_pipeline.py' first to generate base inputs!")
        return
        
    X_base = np.load(X_path)
    y_labels = np.load(y_path)
    print(f"[+] Successfully loaded matrix. Dimensions: {X_base.shape}")

    # Build advanced chemometrics pipelines
    prep_pipelines = {
        "Baseline Only": X_base,
        "Baseline + SNV": apply_snv(X_base),
        "Baseline + MSC": apply_msc(X_base),
        "Baseline + SG Smoothing (W=15, P=2)": savgol_filter(X_base, window_length=15, polyorder=2, deriv=0, axis=1),
        "Baseline + SNV + SG Smoothing": savgol_filter(apply_snv(X_base), window_length=15, polyorder=2, deriv=0, axis=1),
        "Baseline + SG 1st Deriv (W=11, P=2)": savgol_filter(X_base, window_length=11, polyorder=2, deriv=1, axis=1),
        "Baseline + SG 2nd Deriv (W=11, P=3)": savgol_filter(X_base, window_length=11, polyorder=3, deriv=2, axis=1)
    }

    optimization_results = []
    
    print("\n[->] Screening Advanced Preprocessing Combinations...")
    for name, X_processed in prep_pipelines.items():
        print(f"    Evaluating Pipeline Strategy: {name}")
        acc, f1, sen, spec, prec = evaluate_pipeline(X_processed, y_labels)
        
        # Your strict optimization metric calculation
        score = (0.40 * acc) + (0.25 * f1) + (0.15 * sen) + (0.15 * spec) + (0.05 * prec)
        
        optimization_results.append({
            "Pipeline_Combination": name, "Accuracy": acc, "F1_Macro": f1,
            "Sensitivity": sen, "Specificity": spec, "Precision": prec, "Optimization_Score": score
        })

    # Sort and rank all strategies
    df_results = pd.DataFrame(optimization_results).sort_values(by="Optimization_Score", ascending=False)
    df_results.to_csv(os.path.join(RES_DIR, "preprocessing_optimization_results.csv"), index=False)
    
    best_pipeline = df_results.iloc[0]["Pipeline_Combination"]
    print(f"\n[+] Preprocessing Agent Optimization Complete!")
    print(f"    Supreme Combination: {best_pipeline}")
    print(f"    Highest Composite Score: {df_results.iloc[0]['Optimization_Score']:.4f}")

    # Save ultimate feature engine matrix for downstream machine learning
    np.save(os.path.join(RES_DIR, "X_final_chemometrics.npy"), prep_pipelines[best_pipeline])
    print("[*] Locked Matrix saved to 'results/X_final_chemometrics.npy'.")

if __name__ == "__main__":
    main()