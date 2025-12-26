"""
Basketbol olay sınıflandırıcı model
Feature'ları kullanarak basket ve pas olaylarını sınıflandırır
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pandas as pd


class EventClassifier:
    """Basketbol olay sınıflandırıcı"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Args:
            model_type: Model tipi ("random_forest", "xgboost", "svm")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.classes = ["basket", "pas"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def _create_model(self):
        """Model oluştur"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight="balanced"  # Sınıf dengesizliği için
            )
        elif self.model_type == "gradient_boosting":
            # Optimize edilmiş parametreler (basketbol verisi için)
            self.model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")
    
    def _prepare_features(self, features: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature'ları model için hazırla
        
        Args:
            features: Feature dictionary listesi
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
        """
        # Feature'ları DataFrame'e çevir
        df = pd.DataFrame(features)
        
        # Event type'ı label'a çevir
        df['label'] = df['event_type'].map(self.class_to_idx)
        
        # Sınıflandırma için kullanılmayacak kolonları çıkar
        exclude_cols = ['event_type', 'label', 'start_time', 'end_time', 
                       'ball_y_trend', 'has_ball_data']  # Categorical veya metadata
        
        # Numeric kolonları al
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        # NaN değerleri 0 ile doldur
        df[feature_cols] = df[feature_cols].fillna(0)
        
        X = df[feature_cols].values
        y = df['label'].values
        
        return X, y
    
    def train(self, features_file: str, test_size: float = 0.2, random_state: int = 42):
        """
        Model eğit
        
        Args:
            features_file: Feature JSON dosyası
            test_size: Test set oranı
            random_state: Random seed
        """
        print(f"Model eğitimi başlatılıyor...")
        print(f"Model tipi: {self.model_type}")
        print(f"Dosya: {features_file}")
        
        # Feature'ları yükle
        with open(features_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data['features']
        print(f"\nToplam örnek: {len(features)}")
        
        # Event dağılımı
        event_counts = {}
        for f in features:
            event_type = f['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        print("Event dağılımı:")
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}")
        
        # Feature'ları hazırla
        X, y = self._prepare_features(features)
        print(f"\nFeature sayısı: {X.shape[1]}")
        print(f"Feature kolonları: {self.feature_columns[:5]}... ({len(self.feature_columns)} toplam)")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]} örnek")
        print(f"Test set: {X_test.shape[0]} örnek")
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model oluştur ve eğit
        self._create_model()
        
        print(f"\nModel eğitiliyor...")
        self.model.fit(X_train_scaled, y_train)
        
        # Test
        y_pred = self.model.predict(X_test_scaled)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{'='*60}")
        print("TEST SONUÇLARI")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Confusion matrix - gerçek sınıfları al
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        actual_class_names = [self.idx_to_class[idx] for idx in unique_classes]
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        
        print(f"\nConfusion Matrix:")
        print("            Predicted")
        print("            ", end="")
        for cls in actual_class_names:
            print(f"{cls:>8}", end="")
        print()
        for i, cls_idx in enumerate(unique_classes):
            cls_name = self.idx_to_class[cls_idx]
            print(f"Actual {cls_name:>5}  ", end="")
            for j in range(len(unique_classes)):
                print(f"{cm[i][j]:>8}", end="")
            print()
        
        # Classification report - gerçek sınıfları kullan
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  labels=unique_classes,
                                  target_names=actual_class_names, 
                                  zero_division=0))
        
        # Cross-validation
        print(f"\nCross-validation (5-fold):")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'cv_scores': cv_scores.tolist()
        }
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """
        Tek bir feature için tahmin yap
        
        Args:
            features: Feature dictionary
            
        Returns:
            (predicted_class, confidence)
        """
        if self.model is None:
            raise ValueError("Model eğitilmemiş! Önce train() çağırın.")
        
        # Feature'ı vektöre çevir
        feature_vector = np.array([[features.get(col, 0) for col in self.feature_columns]])
        feature_vector = feature_vector.astype(float)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        # Scale
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Tahmin
        pred_proba = self.model.predict_proba(feature_scaled)[0]
        pred_class_idx = self.model.predict(feature_scaled)[0]
        confidence = pred_proba[pred_class_idx]
        
        predicted_class = self.idx_to_class[pred_class_idx]
        
        return predicted_class, confidence
    
    def save(self, model_path: str):
        """Modeli kaydet"""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'model_type': self.model_type
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model kaydedildi: {model_path}")
    
    @classmethod
    def load(cls, model_path: str) -> 'EventClassifier':
        """Modeli yükle"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(model_type=model_data['model_type'])
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.feature_columns = model_data['feature_columns']
        classifier.classes = model_data['classes']
        classifier.class_to_idx = model_data['class_to_idx']
        classifier.idx_to_class = model_data['idx_to_class']
        
        return classifier

