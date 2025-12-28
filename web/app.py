"""
Web Arayüzü - Flask Uygulaması
Basketbol video analizi, kırpma ve sosyal medya paylaşım sistemi

Bu modül, kullanıcılara web arayüzü üzerinden basketbol videolarını yükleyip
analiz etme, tespit edilen olayları (basket, pas) kırpma ve YouTube/Facebook'a
paylaşma imkanı sağlar.

Ana Özellikler:
- Video yükleme ve önizleme
- ML model ile otomatik olay tespiti (basket, pas)
- Tespit edilen olayları video olarak kırpma
- Kırpılmış videoları YouTube/Facebook'a yükleme
- Analiz sonuçlarını cache'leme (hızlı tekrar erişim)

Kullanım:
    cd web
    python app.py
    
Tarayıcıda: http://localhost:5000
"""

import os
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.event_classifier import EventClassifier

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask uygulaması
web_dir = Path(__file__).parent
app = Flask(__name__, 
            template_folder=str(web_dir / 'templates'),
            static_folder=str(web_dir / 'static'))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
CORS(app)

# Klasörler (web dizinine göre)
UPLOAD_FOLDER = web_dir / 'static' / 'uploads'
CLIPS_FOLDER = web_dir / 'clips'
RESULTS_FOLDER = web_dir / 'results'
ANALYSIS_CACHE_FOLDER = web_dir / 'data' / 'analysis_cache'
ANALYSIS_PROGRESS_FOLDER = web_dir / 'data' / 'analysis_progress'
MODEL_PATH = project_root / 'data' / 'models' / 'event_classifier_regularized.pkl'

# Klasörleri oluştur
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CLIPS_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
ANALYSIS_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
ANALYSIS_PROGRESS_FOLDER.mkdir(parents=True, exist_ok=True)

# İzin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Model yükleme (global)
model = None
try:
    if MODEL_PATH.exists():
        model = EventClassifier.load(str(MODEL_PATH))
        logger.info(f"Model yüklendi: {MODEL_PATH}")
    else:
        logger.warning(f"Model bulunamadı: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Model yükleme hatası: {e}")


def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    """
    Video yükleme endpoint'i
    
    GET: Yükleme sayfasını gösterir
    POST: Video dosyasını alır, kaydeder ve analiz sayfasına yönlendirir
    
    Dosya validasyonu:
    - Sadece belirli formatlar (mp4, avi, mov, mkv, webm)
    - Maksimum 500MB
    - Dosya adı timestamp ile güvenli hale getirilir
    """
    if request.method == 'GET':
        return render_template('upload.html')
    
    if 'video' not in request.files:
        return jsonify({'error': 'Video dosyası bulunamadı'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Geçersiz dosya formatı'}), 400
    
    # Dosyayı kaydet
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    filepath = UPLOAD_FOLDER / filename
    file.save(str(filepath))
    
    logger.info(f"Video yüklendi: {filepath}")
    
    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': str(filepath),
        'analyze_url': url_for('analyze_video', filename=filename)
    })


@app.route('/analyze/<filename>')
def analyze_video(filename):
    """Video analizi sayfası"""
    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    return render_template('analyze.html', filename=filename)


def update_progress(filename: str, progress: int, message: str):
    """
    Analiz ilerleme durumunu JSON dosyasına kaydeder
    
    Frontend tarafında polling ile bu dosya okunarak progress bar güncellenir.
    Progress 0-100 arası değer alır, -1 ise hata durumunu gösterir.
    """
    progress_file = ANALYSIS_PROGRESS_FOLDER / f"{filename}_progress.json"
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'progress': progress,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }, f)
    except Exception as e:
        logger.warning(f"Progress kaydetme hatası: {e}")


def analyze_video_in_background(filename: str, filepath: Path, cache_filepath: Path):
    """
    Arka planda video analizini thread içinde çalıştırır
    
    Uzun süren analiz işlemi için ayrı thread kullanılır, böylece Flask
    uygulaması bloke olmaz. İlerleme durumu JSON dosyasına yazılır ve
    sonuçlar cache'e kaydedilir.
    """
    try:
        update_progress(filename, 5, "Video analizi başlatılıyor...")
        
        from web.video_analyzer import get_video_analysis_events_simple
        
        # İlerleme callback'i ile analiz yap
        def progress_callback(step: str, progress: int, message: str):
            update_progress(filename, progress, message)
        
        events = get_video_analysis_events_simple(
            filepath, MODEL_PATH, progress_callback
        )
        
        # Sonuçları cache'e kaydet
        update_progress(filename, 95, "Sonuçlar kaydediliyor...")
        cache_data = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'events': events
        }
        with open(cache_filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        update_progress(filename, 100, "Analiz tamamlandı!")
        logger.info(f"Analiz sonuçları cache'e kaydedildi: {cache_filepath}")
        
    except Exception as e:
        logger.error(f"Arka plan analiz hatası: {e}", exc_info=True)
        update_progress(filename, -1, f"Hata: {str(e)}")


@app.route('/api/analyze', methods=['POST'])
def api_analyze_video():
    """
    Video analizi API endpoint'i - Cache mekanizması ile
    
    İş Akışı:
    1. Cache kontrolü: Eğer daha önce analiz yapıldıysa cache'den döndür
    2. Progress kontrolü: Analiz devam ediyorsa progress bilgisi döndür
    3. Yeni analiz: Cache yoksa arka planda analiz başlat
    
    Cache sayesinde aynı video tekrar analiz edilmez, performans artar.
    """
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'Filename gerekli'}), 400
    
    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    # Cache dosya yolu
    cache_filename = f"{filename}_events.json"
    cache_filepath = ANALYSIS_CACHE_FOLDER / cache_filename
    
    # Cache kontrolü - eğer analiz daha önce yapıldıysa cache'den döndür
    if cache_filepath.exists():
        try:
            logger.info(f"Analiz cache'den yükleniyor: {cache_filepath}")
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return jsonify({
                    'success': True,
                    'events': cached_data.get('events', []),
                    'cached': True
                })
        except Exception as e:
            logger.warning(f"Cache okuma hatası, yeniden analiz yapılacak: {e}")
    
    # Progress dosyası kontrolü - analiz devam ediyorsa
    progress_file = ANALYSIS_PROGRESS_FOLDER / f"{filename}_progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                if progress_data.get('progress', 0) < 100:
                    return jsonify({
                        'success': False,
                        'in_progress': True,
                        'progress': progress_data.get('progress', 0),
                        'message': progress_data.get('message', 'Analiz devam ediyor...')
                    })
        except Exception:
            pass
    
    # Cache yoksa ve analiz başlamamışsa arka planda başlat
    try:
        logger.info(f"Yeni analiz başlatılıyor (arka plan): {filepath}")
        thread = threading.Thread(
            target=analyze_video_in_background,
            args=(filename, filepath, cache_filepath),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'success': False,
            'in_progress': True,
            'progress': 0,
            'message': 'Analiz başlatılıyor...'
        })
    except Exception as e:
        logger.error(f"Analiz başlatma hatası: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/progress/<filename>', methods=['GET'])
def api_get_analyze_progress(filename):
    """Analiz ilerleme durumunu döndür"""
    progress_file = ANALYSIS_PROGRESS_FOLDER / f"{filename}_progress.json"
    
    if not progress_file.exists():
        return jsonify({
            'in_progress': False,
            'progress': 0,
            'message': 'Analiz başlamadı'
        })
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
            progress = progress_data.get('progress', 0)
            
            # Analiz tamamlandıysa cache'den sonuçları kontrol et
            if progress >= 100:
                cache_file = ANALYSIS_CACHE_FOLDER / f"{filename}_events.json"
                if cache_file.exists():
                    with open(cache_file, 'r', encoding='utf-8') as cf:
                        cached_data = json.load(cf)
                        return jsonify({
                            'success': True,
                            'in_progress': False,
                            'progress': 100,
                            'message': 'Analiz tamamlandı',
                            'events': cached_data.get('events', [])
                        })
            
            return jsonify({
                'in_progress': True,
                'progress': progress,
                'message': progress_data.get('message', 'Analiz devam ediyor...')
            })
    except Exception as e:
        logger.error(f"Progress okuma hatası: {e}")
        return jsonify({
            'in_progress': False,
            'progress': 0,
            'message': 'Hata oluştu'
        }), 500


@app.route('/api/clip', methods=['POST'])
def api_clip_video():
    """Video kırpma API endpoint'i"""
    data = request.get_json()
    filename = data.get('filename')
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    event_type = data.get('event_type', 'event')
    
    if not all([filename, start_time is not None, end_time is not None]):
        return jsonify({'error': 'Eksik parametreler'}), 400
    
    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    try:
        from web.video_clipper import clip_video, create_clip_filename
        
        # Çıktı dosya adı oluştur
        clip_filename = create_clip_filename(filename, event_type, start_time, end_time)
        output_path = CLIPS_FOLDER / clip_filename
        
        # Video kırp
        clip_video(filepath, start_time, end_time, output_path)
        
        return jsonify({
            'success': True,
            'clip_filename': clip_filename,
            'download_url': url_for('download_clip', filename=clip_filename)
        })
    except Exception as e:
        logger.error(f"Kırpma hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clip/all', methods=['POST'])
def api_clip_all_events():
    """Tüm olayları kırpma API endpoint'i"""
    data = request.get_json()
    filename = data.get('filename')
    events = data.get('events', [])
    
    if not filename:
        return jsonify({'error': 'Filename gerekli'}), 400
    
    if not events:
        return jsonify({'error': 'Olay listesi boş'}), 400
    
    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    try:
        from web.video_clipper import clip_video, create_clip_filename
        
        clipped_files = []
        
        for i, event in enumerate(events):
            start_time = event.get('start_time')
            end_time = event.get('end_time')
            event_type = event.get('type', 'event')
            
            if start_time is None or end_time is None:
                logger.warning(f"Olay {i+1} için eksik zaman bilgisi, atlanıyor")
                continue
            
            try:
                # Çıktı dosya adı oluştur
                clip_filename = create_clip_filename(filename, event_type, start_time, end_time)
                output_path = CLIPS_FOLDER / clip_filename
                
                # Video kırp
                clip_video(filepath, start_time, end_time, output_path)
                
                clipped_files.append({
                    'clip_filename': clip_filename,
                    'event_type': event_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'download_url': url_for('download_clip', filename=clip_filename)
                })
                
                logger.info(f"Olay {i+1}/{len(events)} kırpıldı: {clip_filename}")
            except Exception as e:
                logger.error(f"Olay {i+1} kırpma hatası: {e}")
                # Hata olsa bile diğer olayları kırpmaya devam et
                continue
        
        return jsonify({
            'success': True,
            'total_events': len(events),
            'clipped_count': len(clipped_files),
            'clips': clipped_files
        })
    except Exception as e:
        logger.error(f"Toplu kırpma hatası: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/download/<filename>')
def download_clip(filename):
    """Kırpılmış video indirme"""
    clip_path = CLIPS_FOLDER / filename
    if not clip_path.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    return send_file(str(clip_path), as_attachment=True, mimetype='video/mp4')


@app.route('/api/clips/stream/<filename>')
def stream_clip(filename):
    """Kırpılmış video streaming (tarayıcıda oynatma için)"""
    clip_path = CLIPS_FOLDER / filename
    if not clip_path.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    return send_file(str(clip_path), mimetype='video/mp4')


@app.route('/results/<filename>')
def results(filename):
    """Sonuçlar sayfası"""
    return render_template('results.html', filename=filename)


@app.route('/api/youtube/upload', methods=['POST'])
def api_youtube_upload():
    """YouTube'a video yükleme API endpoint'i"""
    data = request.get_json()
    clip_filename = data.get('clip_filename')
    
    if not clip_filename:
        return jsonify({'error': 'clip_filename gerekli'}), 400
    
    clip_path = CLIPS_FOLDER / clip_filename
    if not clip_path.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    try:
        from web.youtube_uploader import upload_video_to_youtube
        
        # Video bilgilerini dosya adından çıkar
        parts = Path(clip_filename).stem.split('_')
        event_type = parts[-3] if len(parts) >= 3 else 'basket'
        event_type_label = 'Basket' if event_type == 'basket' else 'Pas'
        
        title = f"{event_type_label} Anı - Basketbol Highlights"
        description = f"Basketbol video analizi ile tespit edilen {event_type_label.lower()} anı."
        tags = ['basketbol', 'basketball', 'highlights', event_type]
        
        result = upload_video_to_youtube(
            video_path=clip_path,
            title=title,
            description=description,
            tags=tags,
            privacy_status='unlisted'  # Varsayılan olarak "unlisted"
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"YouTube yükleme hatası: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/facebook/upload', methods=['POST'])
def api_facebook_upload():
    """Facebook'a video yükleme API endpoint'i"""
    data = request.get_json()
    clip_filename = data.get('clip_filename')
    
    if not clip_filename:
        return jsonify({'error': 'clip_filename gerekli'}), 400
    
    clip_path = CLIPS_FOLDER / clip_filename
    if not clip_path.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    try:
        from web.facebook_uploader import upload_video_to_facebook
        
        # Video bilgilerini dosya adından çıkar
        parts = Path(clip_filename).stem.split('_')
        event_type = parts[-3] if len(parts) >= 3 else 'basket'
        event_type_label = 'Basket' if event_type == 'basket' else 'Pas'
        
        description = f"🏀 {event_type_label} anı - Basketbol highlights\n#basketbol #basketball #highlights"
        
        result = upload_video_to_facebook(
            video_path=clip_path,
            description=description
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Facebook yükleme hatası: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clips/list/<filename>', methods=['GET'])
def api_get_clips(filename):
    """Belirli bir video için kırpılmış videoları listele"""
    try:
        clips = []
        
        # CLIPS_FOLDER'da bu video için kırpılmış dosyaları bul
        base_name = Path(filename).stem
        
        for clip_file in CLIPS_FOLDER.glob(f"{base_name}_*"):
            if clip_file.is_file() and clip_file.suffix in ['.mp4', '.avi', '.mov']:
                # Dosya adından bilgileri çıkar: {base}_{type}_{start}_{end}.mp4
                parts = clip_file.stem.split('_')
                if len(parts) >= 4:
                    try:
                        event_type = parts[-3]  # basket veya pas
                        start_time = float(parts[-2])
                        end_time = float(parts[-1])
                        
                        # Video süresini al
                        import cv2
                        cap = cv2.VideoCapture(str(clip_file))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        duration = frame_count / fps if fps > 0 else (end_time - start_time)
                        cap.release()
                        
                        clips.append({
                            'filename': clip_file.name,
                            'type': event_type,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration
                        })
                    except (ValueError, IndexError):
                        logger.warning(f"Kırpılmış video dosya adı parse edilemedi: {clip_file.name}")
                        continue
        
        # Zaman sırasına göre sırala
        clips.sort(key=lambda x: x['start_time'])
        
        return jsonify({
            'success': True,
            'clips': clips
        })
    except Exception as e:
        logger.error(f"Kırpılmış videolar listelenirken hata: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("Basketbol Video Analizi Web Arayüzü")
    print("="*60)
    print(f"Ana sayfa: http://localhost:5000")
    print(f"Upload klasörü: {UPLOAD_FOLDER}")
    print(f"Clips klasörü: {CLIPS_FOLDER}")
    print("="*60)
    print("\nUygulama başlatılıyor...\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

