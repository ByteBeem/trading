from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = load_model('crypto_predict.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json 
   
    prediction = model.predict(data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)  