import os
import secrets
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Flask Configuration
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

# Folders
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# ✅ SUPPORT TIFF NOW
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Load Model
MODEL_PATH = "C:/Users/chinm/OneDrive/Desktop/LTS-Segnet - Revised/segnet_model.keras"

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Prediction
def predict(image_array):
    global model
    if model is None:
        raise RuntimeError("❌ Model is not loaded!")

    print(f"✅ Input shape: {image_array.shape}")
    start_time = time.time()

    # ✅ FIXED: Direct prediction
    results = model.predict(image_array)

    print(f"⏱️ Inference Time: {time.time() - start_time:.2f}s")
    return results[0]

# DB Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

with app.app_context():
    db.create_all()

# ✅ IMAGE PREPROCESSING (NO DICOM)
def process_xray(file_path):
    image = Image.open(file_path).convert("L")
    image = np.array(image).astype(np.float32)

    # Normalize
    image_min = np.min(image)
    image_max = np.max(image)

    if image_max - image_min > 0:
        image = (image - image_min) / (image_max - image_min)

    # ✅ FIXED SIZE
    image = cv2.resize(image, (128, 128))

    # ✅ FIXED SHAPE
    image = image.reshape(1, 128, 128, 1)

    return image

# Postprocess mask
def postprocess_mask(mask, filename):
    mask = np.squeeze(mask)

    print("Mask min:", np.min(mask), "max:", np.max(mask))  # debug

    # ✅ LOWER THRESHOLD
    mask = (mask > 0.2).astype(np.uint8)

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # ✅ STRONG RED
    colored_mask[mask == 1] = [255, 0, 0]

    output_filename = os.path.splitext(filename)[0] + '_segmentation.png'
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)

    Image.fromarray(colored_mask).save(output_path)
    return output_filename
# Overlay
def stack_images(original_path, segmented_path, output_filename):
    original = Image.open(original_path).convert("RGB")
    segmented = Image.open(segmented_path).convert("RGB")

    # ✅ FIX: match sizes
    segmented = segmented.resize(original.size)

    original_np = np.array(original).astype(np.float32) / 255.0
    segmented_np = np.array(segmented).astype(np.float32) / 255.0

    stacked_np = cv2.addWeighted(original_np, 0.7, segmented_np, 0.3, 0)
    stacked_image = Image.fromarray((stacked_np * 255).astype(np.uint8))

    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    stacked_image.save(output_path)

    return output_filename

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('❌ Username already exists!', 'danger')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('❌ Email already exists!', 'danger')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('✅ Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('❌ Invalid credentials', 'danger')
            return redirect(url_for('login'))
        session['logged_in'] = True
        session['user_id'] = user.id
        flash('✅ Login successful!', 'success')
        return redirect(url_for('upload'))
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('❌ No account associated with this email.', 'danger')
            return redirect(url_for('forgot_password'))

        flash('✅ Password reset email sent. Please check your inbox.', 'success')
        return redirect(url_for('reset_password'))

    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form['email']
        new_password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"success": False, "message": "❌ No account associated with this email."})

        user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.session.commit()
        return jsonify({"success": True, "message": "✅ Password successfully changed!"})

    return render_template('reset_password.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('✅ You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        flash('❌ Please log in first!', 'warning')
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('❌ No file uploaded.', 'danger')
            return redirect(url_for('upload'))
        file = request.files['file']
        if file.filename == '':
            flash('❌ No file selected.', 'danger')
            return redirect(url_for('upload'))
        if not allowed_file(file.filename):
            flash('❌ Only PNG/JPG/TIFF images are allowed!', 'danger')
            return redirect(url_for('upload'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Preprocess & Predict Segmentation
        try:
            preprocessed_image = process_xray(file_path)
            start_time = time.time()
            predicted_mask = predict(preprocessed_image)
            duration = round(time.time() - start_time, 3)  # In seconds
            print(f"✅ Model Inference Time: {duration}s")
            predicted_mask = np.array(predicted_mask)
        except Exception as e:
            print(f'❌ Prediction error: {e}')
            flash('❌ Processing failed.', 'danger')
            return redirect(url_for('upload'))

        if preprocessed_image is None:
            flash('❌ Processing failed.', 'danger')
            return redirect(url_for('upload'))

        segmented_filename = postprocess_mask(predicted_mask, filename)

        # ✅ FIXED: replace dicom_to_png
        original_filename = os.path.splitext(filename)[0] + '_original.png'
        original_path = os.path.join(PROCESSED_FOLDER, original_filename)

        Image.open(file_path).convert("RGB").save(original_path)

        stacked_filename = stack_images(
            original_path,
            os.path.join(PROCESSED_FOLDER, segmented_filename),
            os.path.splitext(filename)[0] + '_stacked.png'
        )

        flash('✅ Image processed successfully!', 'success')

        return redirect(url_for('result', stacked_filename=stacked_filename))

    return render_template('upload.html') 

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/result')
def result():
    stacked_filename = request.args.get('stacked_filename')
    if not stacked_filename:
        flash('❌ No file found for processing!', 'danger')
        return redirect(url_for('upload'))
    stacked_path = url_for('processed_file', filename=stacked_filename)
    return render_template('result.html', stacked_image_url=stacked_path)

# Run App
if __name__ == '__main__':
    app.run(debug=True)