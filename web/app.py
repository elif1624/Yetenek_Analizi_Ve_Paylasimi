"""
Web Arayüzü - Flask Uygulaması
Basketbol video analizi, kırpma ve paylaşım sistemi
"""

import os
import json
import logging
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
MODEL_PATH = project_root / 'data' / 'models' / 'event_classifier.pkl'

# Klasörleri oluştur
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CLIPS_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

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
    """Video yükleme"""
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


@app.route('/api/analyze', methods=['POST'])
def api_analyze_video():
    """Video analizi API endpoint'i"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'Filename gerekli'}), 400
    
    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    try:
        # Video analizi
        from web.video_analyzer import get_video_analysis_events_simple
        
        events = get_video_analysis_events_simple(filepath, MODEL_PATH)
        
        return jsonify({
            'success': True,
            'events': events
        })
    except Exception as e:
        logger.error(f"Analiz hatası: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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


@app.route('/api/instagram/upload', methods=['POST'])
def api_instagram_upload():
    """Instagram'a video paylaşım API endpoint'i"""
    data = request.get_json()
    clip_filename = data.get('clip_filename')
    
    if not clip_filename:
        return jsonify({'error': 'clip_filename gerekli'}), 400
    
    clip_path = CLIPS_FOLDER / clip_filename
    if not clip_path.exists():
        return jsonify({'error': 'Video bulunamadı'}), 404
    
    try:
        from web.instagram_uploader import upload_video_to_instagram
        
        # Video bilgilerini dosya adından çıkar
        parts = Path(clip_filename).stem.split('_')
        event_type = parts[-3] if len(parts) >= 3 else 'basket'
        event_type_label = 'Basket' if event_type == 'basket' else 'Pas'
        
        caption = f"🏀 {event_type_label} anı - Basketbol highlights\n#basketbol #basketball #highlights"
        
        result = upload_video_to_instagram(
            video_path=clip_path,
            caption=caption
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Instagram yükleme hatası: {e}", exc_info=True)
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

