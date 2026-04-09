# app.py
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_flower

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    msg = None

    if request.method == 'POST':
        if 'file' not in request.files:
            msg = "No file selected"
        else:
            file = request.files['file']
            if file.filename == '':
                msg = "No file selected"
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure folder exists
                file.save(filepath)

                # Predict
                img_path, prediction = predict_flower(filepath)
                msg = "Prediction successful!"
            else:
                msg = "Invalid file type. Only images allowed."

    return render_template('index.html', prediction=prediction, img_path=img_path, msg=msg)

if __name__ == '__main__':
    app.run(debug=True)