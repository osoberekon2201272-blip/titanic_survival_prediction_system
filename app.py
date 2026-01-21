from flask import Flask, render_template, request, jsonify
from model import TitanicSurvivalModel
import os

app = Flask(__name__)

# Initialize the model
survival_model = TitanicSurvivalModel()

# Load the trained model (or train if not exists)
if not survival_model.load_model():
    print("No trained model found. Training new model...")
    from model import train_and_save_model

    train_and_save_model()
    survival_model.load_model()


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle survival prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features
        pclass = int(data['pclass'])
        sex = data['sex'].lower()
        age = float(data['age'])
        sibsp = int(data['sibsp'])
        parch = int(data['parch'])
        fare = float(data['fare'])
        embarked = data['embarked'].upper()

        # Validate inputs
        if pclass not in [1, 2, 3]:
            return jsonify({
                'success': False,
                'error': 'Passenger class must be 1, 2, or 3'
            })

        if sex not in ['male', 'female']:
            return jsonify({
                'success': False,
                'error': 'Sex must be "male" or "female"'
            })

        if age < 0 or age > 100:
            return jsonify({
                'success': False,
                'error': 'Age must be between 0 and 100'
            })

        if sibsp < 0 or sibsp > 10:
            return jsonify({
                'success': False,
                'error': 'Siblings/Spouses must be between 0 and 10'
            })

        if parch < 0 or parch > 10:
            return jsonify({
                'success': False,
                'error': 'Parents/Children must be between 0 and 10'
            })

        if fare < 0:
            return jsonify({
                'success': False,
                'error': 'Fare must be a positive number'
            })

        if embarked not in ['S', 'C', 'Q']:
            return jsonify({
                'success': False,
                'error': 'Embarked must be S, C, or Q'
            })

        # Make prediction
        prediction, probability = survival_model.predict(
            pclass=pclass,
            sex=sex,
            age=age,
            sibsp=sibsp,
            parch=parch,
            fare=fare,
            embarked=embarked
        )

        # Return result
        return jsonify({
            'success': True,
            'survived': bool(prediction),
            'probability': round(probability * 100, 1),
            'message': 'Would have survived ✓' if prediction == 1 else 'Would not have survived ✗',
            'inputs': {
                'pclass': pclass,
                'sex': sex,
                'age': age,
                'sibsp': sibsp,
                'parch': parch,
                'fare': fare,
                'embarked': embarked
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)