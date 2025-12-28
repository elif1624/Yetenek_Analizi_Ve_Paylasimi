"""
Eğitim ve Doğrulama Grafikleri Oluşturma Aracı
Accuracy ve Loss grafiklerini oluşturur
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def create_accuracy_graph(train_accuracies, val_accuracies, output_path=None):
    """Eğitim ve Doğrulama Doğruluk Grafiği"""
    epochs = range(len(train_accuracies))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, 'b-', label='Eğitim Doğruluğu', linewidth=2)
    plt.plot(epochs, val_accuracies, 'r-', label='Doğrulama Doğruluğu', linewidth=2)
    
    plt.title('Eğitim ve Doğrulama Doğruluk Grafiği', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Doğruluk (%)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([min(min(train_accuracies), min(val_accuracies)) * 0.95 - 5, 
              max(max(train_accuracies), max(val_accuracies)) * 1.05 + 5])
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Doğruluk grafiği kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_loss_graph(train_losses, val_losses, output_path=None):
    """Eğitim ve Doğrulama Kayıp Grafiği"""
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Eğitim Kayıp', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Doğrulama Kayıp', linewidth=2)
    
    plt.title('Eğitim ve Doğrulama Kayıp Grafiği', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Kayıp', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Kayıp grafiği kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()


def get_training_history(classifier, X_train, y_train, X_val, y_val, n_estimators=30):
    """
    GradientBoostingClassifier için staged prediction ile training history oluştur
    """
    from sklearn.ensemble import GradientBoostingClassifier
    
    if not isinstance(classifier.model, GradientBoostingClassifier):
        raise ValueError("Bu script sadece GradientBoostingClassifier için çalışır")
    
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    # Mevcut model parametrelerini al
    model_params = classifier.model.get_params()
    
    print(f"Training history oluşturuluyor ({n_estimators} epoch)...")
    
    for i in range(1, n_estimators + 1):
        # Yeni model oluştur ve i kadar ağaç ile eğit
        temp_model = GradientBoostingClassifier(
            n_estimators=i,
            max_depth=model_params.get('max_depth', 5),
            learning_rate=model_params.get('learning_rate', 0.1),
            min_samples_split=model_params.get('min_samples_split', 5),
            random_state=model_params.get('random_state', 42),
            subsample=model_params.get('subsample', 1.0),
            max_features=model_params.get('max_features', None)
        )
        
        temp_model.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = temp_model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred) * 100
        train_accuracies.append(train_acc)
        
        # Validation accuracy
        val_pred = temp_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred) * 100
        val_accuracies.append(val_acc)
        
        # Training loss (log loss)
        try:
            train_proba = temp_model.predict_proba(X_train)
            train_loss = log_loss(y_train, train_proba)
        except:
            train_loss = 0.0
        train_losses.append(train_loss)
        
        # Validation loss
        try:
            val_proba = temp_model.predict_proba(X_val)
            val_loss = log_loss(y_val, val_proba)
        except:
            val_loss = 0.0
        val_losses.append(val_loss)
        
        if (i % 5 == 0) or (i == n_estimators):
            print(f"  Epoch {i}/{n_estimators}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    return train_accuracies, val_accuracies, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Eğitim ve Doğrulama Grafikleri Oluştur')
    parser.add_argument(
        '--model-path',
        type=str,
        default='data/models/event_classifier.pkl',
        help='Model dosyası yolu'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/dataset/features.json',
        help='Feature dosyası yolu'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Validation split oranı'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Epoch sayısı (n_estimators)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results/metrics_visualization',
        help='Çıktı dizini'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model dosyası yolu (varsayılan: event_classifier.pkl)'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    features_path = Path(args.features)
    
    if not model_path.exists():
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return 1
    
    if not features_path.exists():
        print(f"HATA: Feature dosyası bulunamadı: {features_path}")
        return 1
    
    print("Model ve veriler yükleniyor...")
    # Model yolu belirtilmişse onu kullan
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"HATA: Model dosyası bulunamadı: {model_path}")
            return 1
    classifier = EventClassifier.load(str(model_path))
    
    # Feature'ları yükle
    with open(features_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data['features']
    
    # Feature'ları hazırla
    X, y = classifier._prepare_features(features)
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_split, random_state=args.random_state, stratify=y
    )
    
    # Feature scaling
    X_train_scaled = classifier.scaler.fit_transform(X_train)
    X_val_scaled = classifier.scaler.transform(X_val)
    
    print(f"\nTrain set: {len(X_train_scaled)} örnek")
    print(f"Validation set: {len(X_val_scaled)} örnek")
    
    # Training history oluştur
    train_accuracies, val_accuracies, train_losses, val_losses = get_training_history(
        classifier, X_train_scaled, y_train, X_val_scaled, y_val, n_estimators=args.epochs
    )
    
    # Çıktı dizini oluştur
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Grafikleri oluştur
    accuracy_path = output_dir / "training_validation_accuracy.png"
    loss_path = output_dir / "training_validation_loss.png"
    
    print("\nGrafikler oluşturuluyor...")
    create_accuracy_graph(train_accuracies, val_accuracies, str(accuracy_path))
    create_loss_graph(train_losses, val_losses, str(loss_path))
    
    print(f"\nGrafikler kaydedildi:")
    print(f"  - Doğruluk grafiği: {accuracy_path}")
    print(f"  - Kayıp grafiği: {loss_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

