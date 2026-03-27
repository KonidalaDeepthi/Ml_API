from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Model Running"

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # California housing features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
        features = [
            data['MedInc'],
            data['HouseAge'],
            data['AveRooms'],
            data['AveBedrms'],
            data['Population'],
            data['AveOccup'],
            data['Latitude'],
            data['Longitude']
        ]
        # Convert to numpy array and reshape for prediction
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({
            'input_features': features,
            'predicted_price': float(prediction[0])
        })
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)