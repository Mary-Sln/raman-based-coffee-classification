import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

BASE_DIR = r"F:\AI-Agents\AI-Kaggle\Raman-IR-AI-Agent"
RES_DIR = os.path.join(BASE_DIR, "results")

class SimcaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            n_comp = min(self.n_components, X_c.shape[0] - 1, X_c.shape[1])
            if n_comp < 1: n_comp = 1
            pca = PCA(n_components=n_comp)
            pca.fit(X_c)
            self.models[c] = (pca, X_c.mean(axis=0))
        return self

    def predict(self, X):
        distances = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            pca, mean_vec = self.models[c]
            X_centered = X - mean_vec
            distances[:, idx] = np.sum((X_centered - pca.inverse_transform(pca.transform(X_centered))) ** 2, axis=1)
        return self.classes_[np.argmin(distances, axis=1)]

class AutoTunedPlsdaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_components=15):
        self.max_components = max_components
        self.best_n_components = 2
        self.pls = None
        self.encoder = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.encoder = OneHotEncoder(sparse_output=False)
        Y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))
        
        upper_limit = min(self.max_components, X.shape[0] - 2, X.shape[1])
        best_score = -1
        
        for n in range(2, upper_limit + 1):
            inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            inner_accs = []
            for train_i, val_i in inner_skf.split(X, y):
                pls_inner = PLSRegression(n_components=n)
                pls_inner.fit(X[train_i], Y_encoded[train_i])
                preds = np.argmax(pls_inner.predict(X[val_i]), axis=1)
                inner_accs.append(accuracy_score(np.argmax(Y_encoded[val_i], axis=1), preds))
            
            mean_inner = np.mean(inner_accs)
            if mean_inner > best_score:
                best_score = mean_inner
                self.best_n_components = n
                
        self.pls = PLSRegression(n_components=self.best_n_components)
        self.pls.fit(X, Y_encoded)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.pls.predict(X), axis=1)]

def main():
    print("[+] Classification Agent Initiated (Auto-Tuning Engine Active)...")
    
    # Address of the filtered matrix (reduced to 300 golden channels)
    X_path = os.path.join(RES_DIR, "X_selected_features.npy")
    y_path = os.path.join(RES_DIR, "y_labels.npy")
    
    if not os.path.exists(X_path):
        print("[-] Error: Missing feature files. Please run feature_selection_agent.py first.")
        return
        
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"[+] Ingested Filtered Features Shape: {X.shape} | Labels Shape: {y.shape}")

    models = {
        "Support Vector Machine": SVC(kernel='linear', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "SIMCA (PCA-Based)": SimcaClassifier(n_components=4),
        "PLS-DA (Auto-Tuned)": AutoTunedPlsdaClassifier(max_components=12),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    cv_results = []

    for name, model in models.items():
        print(f"    Running Stratified 5-Fold Evaluation: {name}...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s, sens, specs, precs = [], [], [], [], []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
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
            
        score = (0.40 * np.mean(accs)) + (0.25 * np.mean(f1s)) + (0.15 * np.mean(sens)) + (0.15 * np.mean(specs)) + (0.05 * np.mean(precs))
            
        cv_results.append({
            "Model": name, "Accuracy": np.mean(accs), "F1_Macro": np.mean(f1s),
            "Sensitivity": np.mean(sens), "Specificity": np.mean(specs), "Precision": np.mean(precs), "Composite_Score": score
        })

    df_res = pd.DataFrame(cv_results).sort_values(by="Composite_Score", ascending=False)
    df_res.to_csv(os.path.join(RES_DIR, "classification_results.csv"), index=False)
    print(f"\n[+] Leaderboard Successfully Updated with {X.shape[1]} Features!")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()