"""
Train-test split kontrolü
"""

import json
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

# Veriyi yükle
features_file = Path("data/dataset/features.json")
with open(features_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data['features']
print(f"Toplam örnek: {len(features)}")

# Event dağılımı
event_counts = {}
for f in features:
    event_type = f['event_type']
    event_counts[event_type] = event_counts.get(event_type, 0) + 1

print("\nToplam dağılım:")
for event_type, count in event_counts.items():
    print(f"  {event_type}: {count}")

# Label'ları oluştur
classes = {'basket': 0, 'pas': 1}
y = np.array([classes.get(f['event_type'], 0) for f in features])  # Bilinmeyen event type varsa 0 olarak işaretle
X = np.arange(len(y))

# Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n{'='*60}")
print("TRAIN-TEST SPLIT")
print(f"{'='*60}")
print(f"Train set: {len(X_train)} örnek (80%)")
print(f"Test set: {len(X_test)} örnek (20%)")

# Train set dağılımı
print(f"\nTrain set dağılımı:")
train_dist = {}
for cls_name, cls_idx in classes.items():
    count = np.sum(y_train == cls_idx)
    train_dist[cls_name] = count
    print(f"  {cls_name}: {count}")

# Test set dağılımı
print(f"\nTest set dağılımı:")
test_dist = {}
for cls_name, cls_idx in classes.items():
    count = np.sum(y_test == cls_idx)
    test_dist[cls_name] = count
    print(f"  {cls_name}: {count}")

print(f"\n{'='*60}")
print("ÖNEMLI:")
print("- Model SADECE train set ile eğitiliyor")
print("- Test set SADECE final değerlendirme için kullanılıyor")
print("- Stratified split: Her sınıfın oranı korunuyor")
print(f"{'='*60}")


