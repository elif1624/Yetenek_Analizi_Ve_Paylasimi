"""
Model Hiperparametreleri Tablosu Oluşturma
Modelin kullandığı hiperparametreleri görsel tablo olarak oluşturur
"""

import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def get_model_hyperparameters(classifier):
    """
    Model hiperparametrelerini al
    """
    model = classifier.model
    
    # Model tipine göre hiperparametreleri al
    if hasattr(model, 'get_params'):
        params = model.get_params()
        
        # Türkçe isimler
        param_names_tr = {
            'n_estimators': 'Ağaç Sayısı',
            'max_depth': 'Maksimum Derinlik',
            'learning_rate': 'Öğrenme Hızı',
            'min_samples_split': 'Min Örnek Split',
            'min_samples_leaf': 'Min Örnek Yaprak',
            'subsample': 'Alt Örnek Oranı',
            'max_features': 'Maksimum Özellik Sayısı',
            'loss': 'Kayıp Fonksiyonu',
            'random_state': 'Rastgele Durum',
            'criterion': 'Kriter'
        }
        
        # Önemli parametreleri seç
        important_params = {
            'n_estimators': params.get('n_estimators', 'N/A'),
            'max_depth': params.get('max_depth', 'N/A'),
            'learning_rate': params.get('learning_rate', 'N/A'),
            'min_samples_split': params.get('min_samples_split', 'N/A'),
            'subsample': params.get('subsample', 'N/A'),
            'max_features': params.get('max_features', 'N/A'),
            'loss': params.get('loss', 'deviance')
        }
        
        # Genel bilgiler
        model_type = classifier.model_type
        
        return important_params, param_names_tr, model_type
    else:
        return {}, {}, 'unknown'


def create_hyperparameters_table(classifier, output_path=None):
    """
    Hiperparametreleri görsel tablo olarak oluştur
    """
    params, param_names_tr, model_type = get_model_hyperparameters(classifier)
    
    # Kayıp fonksiyonu Türkçe
    loss_names = {
        'deviance': 'Log-Loss (Cross-Entropy)',
        'exponential': 'Exponential Loss',
        'log_loss': 'Log-Loss'
    }
    
    # Optimizasyon yöntemi
    if model_type == 'gradient_boosting':
        optimization = 'Gradient Boosting'
    elif model_type == 'random_forest':
        optimization = 'Random Forest'
    else:
        optimization = model_type
    
    # Tablo verileri
    table_data = []
    
    # Optimizasyon
    table_data.append(['Optimizasyon', optimization])
    
    # Öğrenme hızı
    lr = params.get('learning_rate', 'N/A')
    table_data.append(['Öğrenme Hızı', str(lr)])
    
    # Kayıp fonksiyonu
    loss = params.get('loss', 'deviance')
    loss_display = loss_names.get(loss, loss)
    table_data.append(['Kayıp Fonksiyonu', loss_display])
    
    # Ağaç sayısı
    n_estimators = params.get('n_estimators', 'N/A')
    table_data.append(['Ağaç Sayısı (n_estimators)', str(n_estimators)])
    
    # Maksimum derinlik
    max_depth = params.get('max_depth', 'N/A')
    table_data.append(['Maksimum Derinlik (max_depth)', str(max_depth)])
    
    # Min samples split
    min_samples_split = params.get('min_samples_split', 'N/A')
    table_data.append(['Min Örnek Split (min_samples_split)', str(min_samples_split)])
    
    # Subsample
    subsample = params.get('subsample', 'N/A')
    if subsample != 'N/A':
        subsample_display = f'{subsample} ({float(subsample)*100:.0f}%)' if isinstance(subsample, (int, float)) else str(subsample)
    else:
        subsample_display = '1.0 (100%)'
    table_data.append(['Alt Örnek Oranı (subsample)', subsample_display])
    
    # Max features
    max_features = params.get('max_features', 'N/A')
    table_data.append(['Maksimum Özellik (max_features)', str(max_features)])
    
    # Tablo oluştur
    fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Tablo başlıkları
    headers = ['Hiperparametre', 'Değer']
    
    # Tablo oluştur
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0.1, 1, 0.9])
    
    # Tablo stilini ayarla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Başlık satırını vurgula
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.08)
    
    # Diğer satırları vurgula (alternatif renkler)
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('#D9D9D9')
            cell.set_height(0.06)
    
    # İlk sütunu biraz daha geniş yap
    table[(0, 0)].set_width(0.6)
    table[(0, 1)].set_width(0.4)
    
    # Başlık
    title_text = f'Tablo X. Model hiperparametreleri.'
    
    plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Hiperparametre tablosu kaydedildi: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Konsola da yazdır
    print("\n" + "="*70)
    print("MODEL HİPERPARAMETRELERİ")
    print("="*70)
    print(f"{'Hiperparametre':<35} {'Değer':<35}")
    print("-"*70)
    for row in table_data:
        print(f"{row[0]:<35} {row[1]:<35}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Model Hiperparametreleri Tablosu Oluştur')
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/event_classifier.pkl',
        help='Model dosyası yolu'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Görsel çıktı dosyası yolu'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return 1
    
    print("Model yükleniyor...")
    classifier = EventClassifier.load(str(model_path))
    
    # Hiperparametre tablosunu oluştur
    output_path = args.output
    if not output_path:
        output_dir = Path("data/results/metrics_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model dosyasına göre isimlendir
        model_name = model_path.stem
        output_path = output_dir / f"hyperparameters_{model_name}.png"
    
    print("\nHiperparametre tablosu oluşturuluyor...")
    create_hyperparameters_table(classifier, str(output_path))
    
    return 0


if __name__ == "__main__":
    exit(main())


