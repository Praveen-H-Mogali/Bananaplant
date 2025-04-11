import os
import time
from flask import Flask, request, render_template, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import cv2

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# User credentials (For demonstration purposes. Use a database in production.)
users = {
    "admin": generate_password_hash("password123"),  # Pre-hashed password
}

# Load models and define paths
MODEL_PATHS = {
    "part_detection": "models/final_part_detection_model.keras",
    "disease_fruit": "models/final_Fruits_detection_model.keras",
    "disease_leaf": "models/final_Leaves_detection_model.keras",
    "disease_stem": "models/final_Stem_detection_model.keras",
}

try:
    part_detection_model = load_model(MODEL_PATHS["part_detection"])
    disease_models = {
        "Fruits": load_model(MODEL_PATHS["disease_fruit"]),
        "Leaves": load_model(MODEL_PATHS["disease_leaf"]),
        "Stem": load_model(MODEL_PATHS["disease_stem"]),
    }
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    part_detection_model = None
    disease_models = {}

# Labels for parts and diseases
part_labels = ["Fruits", "Leaves", "Stem"]
disease_labels = {
    "Fruits": ["Anthracnose", "Banana Streak Virus", "Cigar End Rot", "Healthy Fruit", "Moko Disease Fruit"],
    "Leaves": [
        "Banana Black Sigatoka Disease",
        "Banana Bract Mosaic Virus Disease",
        "Banana Bunchy Top Virus",
        "Banana Insect Pest Disease",
        "Banana Moko Disease",
        "Banana Panama Disease",
        "Healthy Leaf",
    ],
    "Stem": ["Bacterial Soft Rot", "Banana Xanthomonas", "Healthy Stem", "Panama disease"],
}

# Function to preprocess image
def preprocess_image(image_path, target_size=(128, 128)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# User authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Authenticate user
        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# Protect routes
@app.before_request
def require_login():
    allowed_routes = ['login', 'static']
    if request.endpoint not in allowed_routes and 'user' not in session:
        return redirect(url_for('login'))

# Home route
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/elaichi')
def elaichi():
    return render_template('elaichi.html')

@app.route('/detect-page')
def detect_page():
    return render_template('detect.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for('detect_page'))
    
    file = request.files['file']
    if file.filename == '':
        flash("No file selected", "danger")
        return redirect(url_for('detect_page'))
    
    if file and allowed_file(file.filename):
        # Save the file with a unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        print(f"File uploaded: {file_path}")

        try:
            original_image = cv2.imread(file_path)

            original_height, original_width = original_image.shape[:2]

            # Preprocess image and predict part
            img_array = preprocess_image(file_path)
            part_preds = part_detection_model.predict(img_array)
            detected_part = part_labels[np.argmax(part_preds)]

            # Predict disease based on part
            disease_model = disease_models[detected_part]
            disease_preds = disease_model.predict(img_array)
            detected_disease = disease_labels[detected_part][np.argmax(disease_preds)]
            accuracy = np.max(disease_preds) * 100

            # Solution placeholder
            solutions = {
                "Anthracnose": "Remove infected fruits and debris, Apply fungicides like Mancozeb, Store bananas in cool, dry places.",
                "Banana Streak Virus ": "Use virus-free planting material, Control insect vectors like mealybugs, Avoid planting near infected plants.",
                "Cigar End Rot": "Prune damaged fruits, Improve drainage to reduce humidity, Apply copper-based fungicides.",
                "Moko Disease Fruit": "Remove and destroy infected plants, Sterilize tools, Avoid waterlogging and improve sanitation.",
                "Banana Black Sigatoka Disease": "Remove infected leaves, Use resistant varieties, Apply fungicides like Propiconazole.",
                "Banana Bract Mosaic Virus Disease": "Use virus-free planting materials, Control aphid vectors, Remove infected plants.",
                "Banana Bunchy Top Virus": "Use certified disease-free plants, Remove and destroy infected plants, Control aphids using insecticides.",
                "Banana Insect Pest Disease": "Use insect traps, Apply nematicides or bio-control agents, Improve soil drainage.",
                "Banana Moko Disease": "Remove infected plants, Clean tools regularly, Avoid planting in infected areas.",
                "Banana Panama Disease": "Use resistant cultivars, Improve soil drainage, Rotate crops.",
                "Banana Yellow Sigatoka Disease": "Remove infected leaves, Apply fungicides like Mancozeb, Use resistant varieties.",
                "Bacterial Soft Rot": "Remove infected plants, improve drainage, and sterilize tools.",
                "Banana Xanthomonas": "Remove infected plants, disinfect tools, and practice crop sanitation.",
                "Panama disease": "Use resistant varieties, improve soil drainage, and avoid planting in infected soil.",
                # Add other diseases here
            }
            solution = solutions.get(detected_disease, "No solution available for this disease.")

            bbox_normalized = [0.2, 0.3, 0.7, 0.8]  # Example normalized coordinates
            bbox = [
                int(bbox_normalized[0] * original_width),
                int(bbox_normalized[1] * original_height),
                int(bbox_normalized[2] * original_width),
                int(bbox_normalized[3] * original_height),
            ]

            # Draw the detected region on the original image
            marked_image = original_image.copy()
            cv2.rectangle(marked_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)  # Red bounding box
            cv2.putText(marked_image, detected_disease, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Save the marked image
            marked_file_path = os.path.join(UPLOAD_FOLDER, f"marked_{filename}")
            cv2.imwrite(marked_file_path, marked_image)


            flash(f"Detection complete! Disease: {detected_disease} with {accuracy:.2f}% accuracy.", "success")
            return render_template('detect.html', result={
                "part": detected_part,
                "disease": detected_disease,
                "accuracy": round(accuracy, 2),
                "solution": solution,
                "image_url": file_path,
                "marked_image_url": marked_file_path
            })
        except Exception as e:
            print(f"Error during detection: {e}")
            flash("An error occurred during prediction.", "danger")
            return redirect(url_for('detect_page'))
    
    flash("Invalid file type. Please upload a valid image.", "danger")
    return redirect(url_for('detect_page'))

@app.route('/diseases')
def diseases():
    return render_template('diseases.html', disease_data=disease_labels)

@app.route('/disease/<disease>')
def disease_detail(disease):
    # Data for each disease
    diseases = {
        "anthracnose": {
            "title": "Anthracnose",
            "description1": " A fungus called Colletotrichum musae. ",
            "description2": " Small, black spots on the fruit that grow larger and form sunken areas with orange spore masses.",
            "description3": " All banana varieties, especially ripe fruits.",
            "solution": " Remove infected fruits and debris, Apply fungicides like Mancozeb, Store bananas in cool, dry places.",
        },
        "banana_streak_virus": {
            "title": " Banana Streak Virus (BSV)",
            "description1": " A virus called Banana Streak Virus.",
            "description2": " Yellow streaks on leaves and fruits, reduced fruit size, and poor fruit quality.",
            "description3": " Most cultivated banana types.",
            "solution": " Use virus-free planting material, Control insect vectors like mealybugs, Avoid planting near infected plants.",
        },
        "cigar_end_rot": {
            "title": " Cigar End Rot",
            "description1": " A fungus called Trachysphaera fructigena.",
            "description2": " Black or brown, dry rot starting at the fruit tip, resembling a burnt cigar.",
            "description3": " Mostly affects immature bananas.",
            "solution": " Prune damaged fruits, Improve drainage to reduce humidity, Apply copper-based fungicides.",
        },
        "moko_disease_fruit": {
            "title": " Moko Disease (Fruit)",
            "description1": " A bacterium called Ralstonia solanacearum.",
            "description2": " Premature fruit ripening, discoloration, and internal rot.",
            "description3": " Common in Cavendish and plantain varieties.",
            "solution": " Remove and destroy infected plants, Sterilize tools, Avoid waterlogging and improve sanitation.",
        },
        "banana_black_sigatoka": {
            "title": " Banana Black Sigatoka Disease",
            "description1": " A fungus called Mycosphaerella fijiensis.",
            "description2": " Dark streaks on leaves that grow into large, black spots, leading to leaf drying.",
            "description3": " Most banana types, especially Cavendish.",
            "solution": " Remove infected leaves, Use resistant varieties, Apply fungicides like Propiconazole.",
        },
         "banana_bract_mosaic": {
            "title": " Banana Bract Mosaic Virus Disease",
            "description1": " A virus called Banana Bract Mosaic Virus (BBrMV).",
            "description2": " Red or pink streaks on leaf midribs and pseudostems, distorted leaves, and reduced growth.",
            "description3": " Mostly plantains and Cavendish.",
            "solution": " Use virus-free planting materials, Control aphid vectors, Remove infected plants.",
        },
         "banana_bunchy_top": {
            "title": " Banana Bunchy Top Virus (BBTV)",
            "description1": " A virus called Banana Bunchy Top Virus.",
            "description2": " Leaves appear narrow, stiff, and bunched at the top of the plant.",
            "description3": " All banana types.",
            "solution": " Use certified disease-free plants, Remove and destroy infected plants, Control aphids using insecticides.",
        },
         "banana_insect_pest": {
            "title": " Banana Insect Pest Disease",
            "description1": " Insects like weevils (Cosmopolites sordidus) or nematodes (Radopholus similis).",
            "description2": " Leaf wilting, yellowing, and stunted growth due to root damage.",
            "description3": " All banana types.",
            "solution": " Use insect traps, Apply nematicides or bio-control agents, Improve soil drainage.",
        },
         "banana_moko_disease": {
            "title": " Banana Moko Disease",
            "description1": " A bacterium called Ralstonia solanacearum.",
            "description2": " Yellowing and wilting of leaves, leaf collapse, and plant death.",
            "description3": " Cavendish and plantains.",
            "solution": " Remove infected plants, Clean tools regularly, Avoid planting in infected areas.",
        },
         "banana_panama_disease": {
            "title": " Banana Panama Disease",
            "description1": " A fungus called Fusarium oxysporum f. sp. cubense.",
            "description2": " Yellowing and wilting of older leaves, with reddish-brown streaks in the pseudostem.",
            "description3": "Susceptible varieties like Gros Michel.",
            "solution": " Use resistant cultivars, Improve soil drainage, Rotate crops.",
        },
         "banana_yellow_sigatoka": {
            "title": " Banana Yellow Sigatoka Disease",
            "description1": " A fungus called Mycosphaerella musicola.",
            "description2": " Yellow spots on leaves that enlarge and turn brown, leading to leaf drying.",
            "description3": " Most banana types.",
            "solution": " Remove infected leaves, Apply fungicides like Mancozeb, Use resistant varieties.",
        },
         "bacterial_soft_rot": {
            "title": " Bacterial Soft Rot",
            "description1": " Bacteria (Erwinia carotovora).",
            "description2": "  Stem becomes soft, water-soaked, and emits a foul smell, leading to plant collapse.",
            "description3": " Most banana varieties, especially those grown in waterlogged areas.",
            "solution": " Remove infected plants, improve drainage, and sterilize tools.",
        },
         "banana_xanthomonas": {
            "title": " Banana Xanthomonas (BXW)",
            "description1": " Bacteria (Xanthomonas campestris pv. musacearum).",
            "description2": " Yellowing and wilting of leaves, stem splitting, and bacterial ooze from cut stems.",
            "description3": " Mostly East African Highland bananas and other commonly cultivated varieties.",
            "solution": " Remove infected plants, disinfect tools, and practice crop sanitation.",
        },
         "panama_disease": {
            "title": " Panama Disease",
            "description1": " Fungus (Fusarium oxysporum f. sp. cubense).",
            "description2": " Wilting and yellowing of leaves, with reddish-brown streaks inside the stem.",
            "description3": " Susceptible varieties like Gros Michel, Cavendish, and plantains.",
            "solution": " Use resistant varieties, improve soil drainage, and avoid planting in infected soil.",
        },
        # Add more diseases here
    }

    # Get the details of the disease
    disease_info = diseases.get(disease)

    if disease_info:
        return render_template('disease_detail.html', disease=disease_info)
    else:
        return "Disease not found", 404

if __name__ == '__main__':
    app.run(debug=True)

