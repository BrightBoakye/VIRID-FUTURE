from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import os
from predict import predict_single

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_files'

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST', 'GET'])
def uploader():
    if request.method == "POST":
        try:
            f = request.files['file']
            
            num = len(os.listdir(app.config['UPLOAD_FOLDER']))
            file_format = f.filename.split('.')[-1]
            f.filename = f"image_{num}.{file_format}"

            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            f.save(file_path)

            predicted_label = predict_single(file_path)
            
            return render_template('uploader.html', message=f"Prediction: {predicted_label}")

        except:
            return render_template('uploader.html', message="Error uploading files!")


if __name__ == '__main__':
    app.run(debug=True)