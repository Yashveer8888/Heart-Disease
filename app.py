from flask import Flask, request, render_template, jsonify
import joblib

# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        features = [
            float(request.form['age']),
            float(request.form['cp']),
            float(request.form['thalach'])
        ]
        # Perform prediction
        prediction = model.predict([features])[0]
        if prediction == 1:
            result = "Sorry, Heart Disease Detected"
        elif prediction == 0:
            result = "No Heart Disease" 
        else:
            result ="sorry"
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
