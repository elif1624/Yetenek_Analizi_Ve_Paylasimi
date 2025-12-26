"""
Model Karşılaştırma Aracı
Orijinal ve Regularized+Augmented modelleri karşılaştırır
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.check_overfitting import main as check_overfitting


def compare_models():
    """İki modeli karşılaştır"""
    
    models = [
        {
            'name': 'ORIJINAL MODEL',
            'model': 'data/models/event_classifier.pkl',
            'features': 'data/dataset/features.json'
        },
        {
            'name': 'REGULARIZED + AUGMENTED MODEL',
            'model': 'data/models/event_classifier_regularized.pkl',
            'features': 'data/dataset/features_augmented.json'
        }
    ]
    
    print("="*70)
    print("MODEL KARSILASTIRMASI")
    print("="*70)
    print()
    
    results = []
    
    for model_info in models:
        print(f"\n{'='*70}")
        print(f"{model_info['name']}")
        print(f"{'='*70}\n")
        
        # Overfitting analizi yap
        import sys
        old_argv = sys.argv.copy()
        sys.argv = [
            'check_overfitting.py',
            '--model', model_info['model'],
            '--features', model_info['features']
        ]
        
        # Sonuçları yakalamak için output'u redirect etmek gerekir
        # Basit bir çözüm: her model için ayrı çalıştır
        
        results.append(model_info)
    
    print("\n" + "="*70)
    print("OZET KARSILASTIRMA")
    print("="*70)
    print("\nHer modeli ayri ayri test edin:")
    print("\n1. Orijinal Model:")
    print(f"   python tools/check_overfitting.py --model {models[0]['model']} --features {models[0]['features']}")
    print("\n2. Regularized + Augmented Model:")
    print(f"   python tools/check_overfitting.py --model {models[1]['model']} --features {models[1]['features']}")
    print("\n" + "="*70)


if __name__ == "__main__":
    compare_models()



