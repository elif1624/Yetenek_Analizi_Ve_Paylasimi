"""
Basketbol olay sınıflandırma için model optimizasyonu
Farklı modelleri test eder ve en iyisini seçer
"""

import json
import argparse
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, LeaveOneOut
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pandas as pd
from collections import defaultdict

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier


class ModelOptimizer:
    """Model optimizasyonu için sınıf"""
    
    def __init__(self, features_file: str, test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            features_file: Feature JSON dosyası
            test_size: Test set oranı
            random_state: Random seed
        """
        self.features_file = features_file
        self.test_size = test_size
        self.random_state = random_state
        self.classes = ["basket", "pas"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Veriyi yükle
        self._load_data()
        
    def _load_data(self):
        """Veriyi yükle ve hazırla"""
        with open(self.features_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data['features']
        
        # DataFrame'e çevir
        df = pd.DataFrame(features)
        df['label'] = df['event_type'].map(self.class_to_idx)
        
        # Feature kolonlarını belirle
        exclude_cols = ['event_type', 'label', 'start_time', 'end_time', 
                       'ball_y_trend', 'has_ball_data']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # NaN değerleri 0 ile doldur
        df[self.feature_columns] = df[self.feature_columns].fillna(0)
        
        X = df[self.feature_columns].values
        y = df['label'].values
        
        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Veri yüklendi: {len(features)} örnek")
        print(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        print(f"Feature sayısı: {len(self.feature_columns)}\n")
    
    def test_random_forest(self):
        """Random Forest modellerini test et"""
        print("=" * 70)
        print("RANDOM FOREST MODEL OPTİMİZASYONU")
        print("=" * 70)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
        
        # Grid search (küçük dataset için Leave-One-Out CV kullan)
        cv = StratifiedKFold(n_splits=min(5, len(self.X_train)), shuffle=True, random_state=42)
        
        rf = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\nEn iyi parametreler: {best_params}")
        print(f"CV Accuracy: {best_score:.3f}")
        
        # Test set ile değerlendir
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test F1-Score: {test_f1:.3f}\n")
        
        return {
            'model': 'RandomForest',
            'best_params': best_params,
            'cv_score': best_score,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'model_instance': best_model
        }
    
    def test_gradient_boosting(self):
        """Gradient Boosting modellerini test et"""
        print("=" * 70)
        print("GRADIENT BOOSTING MODEL OPTİMİZASYONU")
        print("=" * 70)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.2],
            'min_samples_split': [2, 5]
        }
        
        cv = StratifiedKFold(n_splits=min(5, len(self.X_train)), shuffle=True, random_state=42)
        
        gb = GradientBoostingClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            gb, param_grid, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\nEn iyi parametreler: {best_params}")
        print(f"CV Accuracy: {best_score:.3f}")
        
        # Test set ile değerlendir
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test F1-Score: {test_f1:.3f}\n")
        
        return {
            'model': 'GradientBoosting',
            'best_params': best_params,
            'cv_score': best_score,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'model_instance': best_model
        }
    
    def test_svm(self):
        """SVM modellerini test et"""
        print("=" * 70)
        print("SVM MODEL OPTİMİZASYONU")
        print("=" * 70)
        
        # Hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'class_weight': ['balanced', None]
        }
        
        cv = StratifiedKFold(n_splits=min(5, len(self.X_train)), shuffle=True, random_state=42)
        
        svm = SVC(random_state=self.random_state, probability=True)
        grid_search = GridSearchCV(
            svm, param_grid, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\nEn iyi parametreler: {best_params}")
        print(f"CV Accuracy: {best_score:.3f}")
        
        # Test set ile değerlendir
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test F1-Score: {test_f1:.3f}\n")
        
        return {
            'model': 'SVM',
            'best_params': best_params,
            'cv_score': best_score,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'model_instance': best_model
        }
    
    def compare_models(self):
        """Tüm modelleri karşılaştır"""
        print("=" * 70)
        print("MODEL KARŞILAŞTIRMA")
        print("=" * 70)
        
        results = []
        
        # Random Forest
        try:
            rf_result = self.test_random_forest()
            results.append(rf_result)
        except Exception as e:
            print(f"Random Forest hatası: {e}\n")
        
        # Gradient Boosting
        try:
            gb_result = self.test_gradient_boosting()
            results.append(gb_result)
        except Exception as e:
            print(f"Gradient Boosting hatası: {e}\n")
        
        # SVM
        try:
            svm_result = self.test_svm()
            results.append(svm_result)
        except Exception as e:
            print(f"SVM hatası: {e}\n")
        
        # Sonuçları karşılaştır
        print("=" * 70)
        print("KARŞILAŞTIRMA SONUÇLARI")
        print("=" * 70)
        print(f"{'Model':<20} {'CV Score':<12} {'Test Acc':<12} {'Test F1':<12}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['model']:<20} {r['cv_score']:.3f}        {r['test_accuracy']:.3f}        {r['test_f1']:.3f}")
        
        # En iyi modeli seç (CV score'a göre)
        best_result = max(results, key=lambda x: x['cv_score'])
        
        print("\n" + "=" * 70)
        print(f"EN İYİ MODEL: {best_result['model']}")
        print("=" * 70)
        print(f"CV Score: {best_result['cv_score']:.3f}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.3f}")
        print(f"Test F1-Score: {best_result['test_f1']:.3f}")
        print(f"Parametreler: {best_result['best_params']}")
        
        return best_result
    
    def feature_importance_analysis(self, model):
        """Feature importance analizi"""
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALİZİ")
        print("=" * 70)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n{'Feature':<30} {'Importance':<12}")
            print("-" * 70)
            for feature, importance in feature_importance[:15]:  # Top 15
                print(f"{feature:<30} {importance:.4f}")
        else:
            print("Model feature importance desteği yok.")


def main():
    parser = argparse.ArgumentParser(description="Basketbol olay sınıflandırma model optimizasyonu")
    parser.add_argument(
        "--features",
        type=str,
        default="data/dataset/features.json",
        help="Feature dosyası yolu (default: data/dataset/features.json)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set oranı (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Dosya kontrolü
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"HATA: Feature dosyası bulunamadı: {features_path}")
        return 1
    
    # Optimizer oluştur
    optimizer = ModelOptimizer(
        features_file=str(features_path),
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Modelleri karşılaştır
    best_result = optimizer.compare_models()
    
    # Feature importance
    optimizer.feature_importance_analysis(best_result['model_instance'])
    
    print("\n" + "=" * 70)
    print("OPTİMİZASYON TAMAMLANDI")
    print("=" * 70)
    print(f"Önerilen model: {best_result['model']}")
    print(f"Parametreler: {best_result['best_params']}")
    
    return 0


if __name__ == "__main__":
    exit(main())


