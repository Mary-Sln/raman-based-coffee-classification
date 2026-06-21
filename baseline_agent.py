import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import linalg, diags
from scipy.spatial import ConvexHull
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set project path configurations based on flat workspace layout
BASE_DIR = r"F:\AI-Agents\AI-Kaggle\Raman-IR-AI-Agent"
RES_DIR = os.path.join(BASE_DIR, "results")
EXCEL_PATH = os.path.join(BASE_DIR, "coffee_data.xlsx")
os.makedirs(RES_DIR, exist_ok=True)

# --- Mathematical baseline correction functions ---

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L)).tocsr()
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L)).tocsr()
        Z = W + lam * D.T @ D
        z = linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_airpls(y, lam=1e5, niter=20):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L)).tocsr()
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0, shape=(L, L)).tocsr()
        Z = W + lam * D.T @ D
        z = linalg.spsolve(Z, w * y)
        d = y - z
        d_neg = d[d < 0]
        if len(d_neg) == 0:
            break
        w = np.exp(i * np.abs(d) / np.abs(d_neg).sum())
        w[d >= 0] = 0
    return z

def baseline_arpls(y, lam=1e5, ratio=0.01, niter=20):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L)).tocsr()
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L)).tocsr()
        Z = W + lam * D.T @ D
        z = linalg.spsolve(Z, w * y)
        d = y - z
        w = 1.0 / (1.0 + np.exp(2 * (d - d[d<0].mean()) / (d[d<0].std() + 1e-8)))
        w[d >= 0] = ratio
    return z

def baseline_rubberband(y):
    L = len(y)
    x = np.arange(L)
    points = np.vstack((x, y)).T
    augmented = np.vstack([points, [0, max(y)*2], [L-1, max(y)*2]])
    hull = ConvexHull(augmented)
    vertices = sorted([v for v in hull.vertices if v < L])
    return np.interp(x, x[vertices], y[vertices])

def baseline_polynomial(y, order=3):
    x = np.arange(len(y))
    poly_coeffs = np.polyfit(x, y, order)
    return np.polyval(poly_coeffs, x)

# --- Downstream classification evaluation loop ---

def evaluate_baseline_performance(X, y_labels):
    unique_classes, class_counts = np.unique(y_labels, return_counts=True)
    n_splits = 5 if np.min(class_counts) >= 5 else int(np.min(class_counts))
    
    if n_splits < 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1500, random_state=42)
        model.fit(X_scaled, y_labels)
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y_labels, y_pred)
        return acc, acc, acc, acc, acc

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, sens, specs, precs = [], [], [], [], []
    
    for train_idx, test_idx in skf.split(X, y_labels):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        # Smart in-fold standardization without data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1500, random_state=42)
        model.fit(X_train_scaled, y_train) # Training model using scaled in-fold data
        y_pred = model.predict(X_test_scaled)
        
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        sens.append(recall_score(y_test, y_pred, average='macro'))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)  
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specs.append(np.mean(tn / (tn + fp + 1e-8)))

    return np.mean(accs), np.mean(f1s), np.mean(sens), np.mean(specs), np.mean(precs)

def main():
    print("[+] Baseline Correction Agent Started...")
    if not os.path.exists(EXCEL_PATH):
        print(f"[-] Missing: {EXCEL_PATH}")
        return

    excel_file = pd.ExcelFile(EXCEL_PATH)
    global_spectra, labels = [], []

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
        if df.empty or df.shape[1] < 2:
            continue
        df_clean = df.dropna(subset=[df.columns[0]]).iloc[:785, :] 
        samples_df = df_clean.iloc[:, 1:].T
        
        for sample_name, row in samples_df.iterrows():
            global_spectra.append(row.values)
            labels.append(sheet)

    X_cleaned = np.array(global_spectra, dtype=float)
    y_labels = np.array(labels)
    total_samples = len(X_cleaned)
    print(f"[+] Total integrated matrix ready: {X_cleaned.shape}")

    methods = ["No correction", "ALS", "airPLS", "arPLS", "Rubberband", "Polynomial baseline"]
    comparison_results = []
    corrected_matrices = {}

    for method in methods:
        print(f"\n[->] Running Method: {method}")
        X_corr = np.zeros_like(X_cleaned)

        for i in range(total_samples):
            sys.stdout.write(f"\r    Processing Spectrum: {i+1}/{total_samples}")
            sys.stdout.flush()

            if method == "No correction":
                X_corr[i] = X_cleaned[i]
            elif method == "ALS":
                X_corr[i] = X_cleaned[i] - baseline_als(X_cleaned[i])
            elif method == "airPLS":
                X_corr[i] = X_cleaned[i] - baseline_airpls(X_cleaned[i])
            elif method == "arPLS":
                X_corr[i] = X_cleaned[i] - baseline_arpls(X_cleaned[i])
            elif method == "Rubberband":
                X_corr[i] = X_cleaned[i] - baseline_rubberband(X_cleaned[i])
            elif method == "Polynomial baseline":
                X_corr[i] = X_cleaned[i] - baseline_polynomial(X_cleaned[i], order=3)

        print("\n    Testing classification score...")
        corrected_matrices[method] = X_corr
        acc, f1, sen, spec, prec = evaluate_baseline_performance(X_corr, y_labels)
        score = (0.40 * acc) + (0.25 * f1) + (0.15 * sen) + (0.15 * spec) + (0.05 * prec)
        
        comparison_results.append({
            "Method": method, "Accuracy": acc, "F1": f1, 
            "Sensitivity": sen, "Specificity": spec, "Precision": prec, "Composite_Score": score
        })

    df_comp = pd.DataFrame(comparison_results).sort_values(by="Composite_Score", ascending=False)
    df_comp.to_csv(os.path.join(RES_DIR, "baseline_comparison.csv"), index=False)
    
    best_method = df_comp.iloc[0]["Method"]
    print(f"\n\n[+] Evaluation Done! Supreme Winner: {best_method} | Score: {df_comp.iloc[0]['Composite_Score']:.4f}")
    
    np.save(os.path.join(RES_DIR, "X_preprocessed.npy"), corrected_matrices[best_method])
    np.save(os.path.join(RES_DIR, "y_labels.npy"), y_labels)

if __name__ == "__main__":
    main()