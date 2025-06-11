from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from predict import predict_and_annotate

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    pred_class, annotated_path = predict_and_annotate(filepath)
    return jsonify({
        "prediction": pred_class,
        "annotated_image": annotated_path
    })

if __name__ == "__main__":
    app.run(debug=True)