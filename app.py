from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

prediction_pipeline = PredictionPipeline()
config = prediction_pipeline.prediction_config

UPLOAD_FOLDER = config.prediction_img_load_path
PREDICT_FOLDER = config.predicted_img_save_path

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)
        
        files = request.files.getlist('files')  # Çoklu dosya yükleme
        for file in files:
            if file.filename == '':
                continue
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
        return redirect(url_for('upload_file'))
    
    real_images = os.listdir(UPLOAD_FOLDER)
    predict_images = os.listdir(PREDICT_FOLDER)
    return render_template('index.html', real_images=real_images, predict_images=predict_images)

@app.route('/real_images/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/predict_images/<filename>')
def predicted_file(filename):
    return send_from_directory(PREDICT_FOLDER, filename)

@app.route('/delete_real_image/<filename>', methods=['POST'])
def delete_real_image(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('upload_file'))

@app.route('/generate', methods=['POST'])
def generate():
    prediction_pipeline.run_prediction_pipeline()
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
