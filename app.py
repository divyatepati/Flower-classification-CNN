# app.py
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_flower

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

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

                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                file.save(filepath)

                # Prediction
                prediction = predict_flower(filepath)
                img_path = filepath
                msg = "Prediction successful!"

            else:
                msg = "Invalid file type"

    return render_template('index.html',
                           prediction=prediction,
                           img_path=img_path,
                           msg=msg)


# ✅ IMPORTANT FOR RENDER
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)