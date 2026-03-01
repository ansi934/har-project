import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# --- 1. DATA GENERATION (6 Distinct Activities) ---
def generate_data(n_samples=500):
    """
    Generates synthetic 3-axis accelerometer data with DISTINCT patterns.
    Activities: Sitting, Standing, Walking, Jogging, Running, Jumping
    """
    data = []
    labels = []
    
    for _ in range(n_samples):
        
        # --- 1. SITTING ---
        # Very low movement, stable Z near gravity (9.8)
        acc_x = np.random.normal(0, 0.2)
        acc_y = np.random.normal(0, 0.2)
        acc_z = np.random.normal(9.8, 0.1)
        data.append([acc_x, acc_y, acc_z])
        labels.append('Sitting')

        # --- 2. STANDING ---
        # Slight body sway, Z still near gravity
        acc_x = np.random.normal(0.2, 0.5)
        acc_y = np.random.normal(0.2, 0.5)
        acc_z = np.random.normal(9.7, 0.3)
        data.append([acc_x, acc_y, acc_z])
        labels.append('Standing')

        # --- 3. WALKING ---
        # Moderate movement, clear pattern, Z drops slightly
        acc_x = np.random.normal(2.0, 1.0)
        acc_y = np.random.normal(3.0, 1.0)
        acc_z = np.random.normal(9.0, 1.0)
        data.append([acc_x, acc_y, acc_z])
        labels.append('Walking')

        # --- 4. JOGGING ---
        # Higher frequency, more variance than walking
        acc_x = np.random.normal(5.0, 2.0)
        acc_y = np.random.normal(6.0, 2.0)
        acc_z = np.random.normal(8.0, 2.0)
        data.append([acc_x, acc_y, acc_z])
        labels.append('Jogging')

        # --- 8. RUNNING ---
        # High intensity, Z drops significantly (feet not touching much)
        acc_x = np.random.normal(10.0, 3.0)
        acc_y = np.random.normal(12.0, 3.0)
        acc_z = np.random.normal(4.0, 2.5)
        data.append([acc_x, acc_y, acc_z])
        labels.append('Running')

        # --- 6. JUMPING ---
        # SUDDEN upward force, very HIGH Z (key difference!)
        # X and Y are lower than running, but Z spikes up
        acc_x = np.random.normal(1.0, 2.0)
        acc_y = np.random.normal(0.5, 2.0)
        acc_z = np.random.normal(20.0, 5.0)  # Much higher than running!
        data.append([acc_x, acc_y, acc_z])
        labels.append('Jumping')

    return np.array(data), np.array(labels)

# --- 2. MODEL TRAINING ---
def train_model():
    print("Generating dataset with 6 distinct activities...")
    X, y = generate_data()
    
    # Print data distribution
    print(f"Total samples: {len(X)}")
    print(f"Features shape: {X.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest with more trees and better parameters
    print("Training Model...")
    clf = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    clf.fit(X_scaled, y)
    
    # Print training accuracy
    train_accuracy = clf.score(X_scaled, y)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # Save artifacts
    joblib.dump(clf, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model trained and saved!")

# Check if model exists, if not, train it
if not os.path.exists('model.pkl'):
    train_model()

# --- 3. FLASK ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/retrain', methods=['POST'])
def retrain():
    train_model()
    return jsonify({"message": "Model retrained successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x = float(data['x'])
        y = float(data['y'])
        z = float(data['z'])
        
        # Load model and scaler
        clf = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Predict
        features = np.array([[x, y, z]])
        features_scaled = scaler.transform(features)
        prediction = clf.predict(features_scaled)[0]
        
        # Get probability for confidence
        proba = clf.predict_proba(features_scaled)[0]
        classes = clf.classes_
        max_prob = max(proba) * 100
        confidence = f"{max_prob:.1f}%"
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)})   

if __name__ == '__main__':
    app.run(debug=True)