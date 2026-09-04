
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>- WLASL</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            min-height: 100vh;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            max-width: 800px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #0f3460;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-area:hover {
            background: #e8f0fe;
            border-color: #16213e;
        }
        .upload-area.dragover {
            background: #d4e4ff;
            border-color: #0f3460;
        }
        .upload-icon {
            font-size: 50px;
            margin-bottom: 15px;
            display: block;
        }
        input[type="file"] {
            display: none;
        }
        .btn {
            background: #0f3460;
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 30px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 15px;
        }
        .btn:hover {
            background: #1a1a2e;
            transform: scale(1.02);
        }
        .btn:disabled {
            background: #999;
            cursor: not-allowed;
            transform: none;
        }
        .result-box {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            display: none;
            animation: slideIn 0.5s ease;
        }
        .result-box.success {
            background: #e8f5e9;
            border: 2px solid #4caf50;
            display: block;
        }
        .result-box.error {
            background: #ffebee;
            border: 2px solid #f44336;
            display: block;
        }
        .result-box.loading {
            background: #fff3e0;
            border: 2px solid #ff9800;
            display: block;
        }
        .result-label {
            font-weight: bold;
            font-size: 14px;
            color: #555;
            margin-bottom: 5px;
        }
        .result-text {
            font-size: 24px;
            font-weight: bold;
            color: #1a1a2e;
            direction: ltr;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-badge.online { background: #4caf50; color: white; }
        .status-badge.offline { background: #f44336; color: white; }
        .status-badge.loading { background: #ff9800; color: white; }
        .footer {
            margin-top: 20px;
            text-align: center;
            color: #999;
            font-size: 12px;
        }
        .file-name {
            margin-top: 10px;
            color: #0f3460;
            font-weight: bold;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .endpoints {
            margin-top: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 10px;
            font-size: 12px;
        }
        .endpoints code {
            background: #e0e0e0;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
        }
        .video-preview {
            max-width: 100%;
            margin-top: 15px;
            border-radius: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> مترجم لغة الإشارة</h1>
        <p class="subtitle">قم بتحميل فيديو للترجمة إلى نص</p>

        <div class="upload-area" id="uploadArea">
            <span class="upload-icon">📹</span>
            <p>اسحب الفيديو هنا أو اضغط للاختيار</p>
            <input type="file" id="videoInput" accept="video/*">
            <br>
            <button class="btn" id="predictBtn" disabled> ترجمة</button>
            <div class="file-name" id="fileName"></div>
        </div>

        <div id="result" class="result-box">
            <div class="result-label">نتيجة الترجمة:</div>
            <div class="result-text" id="resultText"></div>
        </div>

        <div style="margin-top: 20px; text-align: center;">
            <span class="status-badge loading" id="statusBadge">⏳ جاري التحميل...</span>
        </div>

        <div class="endpoints">
            <strong>📡 نقاط النهاية API:</strong><br>
            <code>POST /predict_video</code> - رفع فيديو للترجمة<br>
            <code>POST /predict_frames</code> - رفع إطارات مشفرة<br>
            <code>GET /health</code> - التحقق من الحالة<br>
            <code>GET /vocabulary</code> - عرض القاموس
        </div>

        <div class="footer">
             WLASL Sign Language Translator v1.0
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const videoInput = document.getElementById('videoInput');
        const predictBtn = document.getElementById('predictBtn');
        const fileName = document.getElementById('fileName');
        const result = document.getElementById('result');
        const resultText = document.getElementById('resultText');
        const statusBadge = document.getElementById('statusBadge');

        // Check model status
        function checkStatus() {
            fetch('/health')
                .then(res => res.json())
                .then(data => {
                    if (data.model_loaded) {
                        statusBadge.textContent = '✅ النموذج جاهز';
                        statusBadge.className = 'status-badge online';
                        predictBtn.disabled = false;
                    } else {
                        statusBadge.textContent = '⏳ جاري تحميل النموذج...';
                        statusBadge.className = 'status-badge loading';
                        predictBtn.disabled = true;
                        setTimeout(checkStatus, 3000);
                    }
                })
                .catch(() => {
                    statusBadge.textContent = '❌ غير متصل';
                    statusBadge.className = 'status-badge offline';
                });
        }

        checkStatus();

        // Upload area events
        uploadArea.addEventListener('click', () => videoInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                videoInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });

        videoInput.addEventListener('change', handleFileSelect);

        function handleFileSelect() {
            if (videoInput.files.length > 0) {
                fileName.textContent = '📁 ' + videoInput.files[0].name;
                predictBtn.disabled = false;
            } else {
                fileName.textContent = '';
                predictBtn.disabled = true;
            }
        }

        predictBtn.addEventListener('click', predictVideo);

        function predictVideo() {
            if (!videoInput.files || videoInput.files.length === 0) {
                alert('الرجاء اختيار فيديو');
                return;
            }

            const file = videoInput.files[0];
            const formData = new FormData();
            formData.append('video', file);

            result.className = 'result-box loading';
            resultText.textContent = 'جاري معالجة الفيديو...';
            predictBtn.disabled = true;

            fetch('/predict_video', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    result.className = 'result-box success';
                    resultText.textContent = data.prediction || '⚠️ لم يتم التعرف على إشارة';
                } else {
                    result.className = 'result-box error';
                    resultText.textContent = '' + (data.error || data.message || 'خطأ غير معروف');
                }
                predictBtn.disabled = false;
            })
            .catch(error => {
                result.className = 'result-box error';
                resultText.textContent = ' خطأ في الاتصال: ' + error.message;
                predictBtn.disabled = false;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
   
    return jsonify({
        'status': 'online',
        'name': 'WLASL Sign Language Translator API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'GET - API information',
            '/test': 'GET - Test page (HTML)',
            '/health': 'GET - Health check',
            '/vocabulary': 'GET - Vocabulary information',
            '/predict_video': 'POST - Upload video for translation',
            '/predict_frames': 'POST - Upload base64 frames',
            '/predict_batch': 'POST - Upload multiple videos'
        }
    })

@app.route('/test')
def test_page():
    """Test page with video upload interface."""
    return HTML_TEST_PAGE

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    global model, model_status
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_status': model_status,
        'device': str(model.device) if model else 'None',
        'vocabulary_size': len(model.vocab) if model else 0,
        'is_trained': model.model is not None if model else False
    })

@app.route('/vocabulary', methods=['GET'])
def get_vocabulary():
    """Get vocabulary information."""
    global model
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 503
    
    words = list(model.vocab.word2idx.keys())
    return jsonify({
        'vocabulary_size': len(words),
        'words': words
    })

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Predict from uploaded video file."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model is still loading. Please try again.'}), 503
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Save video temporarily
    temp_path = f'temp_video_{int(time.time()*1000)}.mp4'
    video_file.save(temp_path)
    
    try:
        # Predict
        result = model.predict(video_path=temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'message': 'Translation completed successfully'
        })
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/predict_frames', methods=['POST'])
def predict_frames():
    """Predict from base64 encoded frames."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model is still loading. Please try again.'}), 503
    
    data = request.json
    if not data or 'frames' not in data:
        return jsonify({'error': 'No frames provided'}), 400
    
    try:
        frames_data = data['frames']
        frames = []
        
        for frame_b64 in frames_data:
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frames.append(frame)
        
        result = model.predict(frames=frames)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'message': 'Translation completed successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict from multiple video files."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model is still loading. Please try again.'}), 503
    
    if 'videos' not in request.files:
        return jsonify({'error': 'No video files provided'}), 400
    
    video_files = request.files.getlist('videos')
    results = []
    
    for video_file in video_files:
        temp_path = f'temp_video_{int(time.time()*1000)}_{video_file.filename}'
        video_file.save(temp_path)
        
        try:
            result = model.predict(video_path=temp_path)
            results.append({
                'filename': video_file.filename,
                'prediction': result
            })
        except Exception as e:
            results.append({
                'filename': video_file.filename,
                'error': str(e)
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return jsonify({
        'success': True,
        'results': results
    })

# =============================================================================
# 10. Model Initialization
# =============================================================================

def init_model():
    """Initialize the model in background."""
    global model, model_status
    
    print("🔧 Initializing model...")
    model_status = "loading"
    
    try:
        # Try to find dataset path
        dataset_path = None
        possible_paths = [
            "/root/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5",
            "wlasl_data"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        model = SignLanguageModel(
            model_path='best_sign_language_model.pt',
            dataset_path=dataset_path
        )
        model_status = "ready"
        print("✅ Model initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        model_status = "error"
        model = None

# =============================================================================
# 11. Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤟 WLASL Sign Language Translator - Backend API")
    print("="*60)
    
    # Start model initialization in background
    thread = threading.Thread(target=init_model)
    thread.start()
    
    # Get host and port
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🌐 Server running at: http://{host}:{port}")
    print(f"🧪 Test page at: http://{host}:{port}/test")
    print("\n📡 API Endpoints:")
    print("   GET  /              - API information")
    print("   GET  /test          - Test page")
    print("   GET  /health        - Health check")
    print("   GET  /vocabulary    - Vocabulary info")
    print("   POST /predict_video - Upload video for translation")
    print("   POST /predict_frames - Upload frames for translation")
    print("   POST /predict_batch - Upload multiple videos")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host=host, port=port, debug=False, threaded=True)