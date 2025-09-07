 
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("rf.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    df = pd.DataFrame([data])

    for col in label_encoders:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])

    if 'Sustainability_Rating' in df.columns:
        df = df.drop('Sustainability_Rating', axis=1)

    prediction = model.predict(df)

    decoded = label_encoders['Sustainability_Rating'].inverse_transform(prediction)

    return jsonify({'prediction': decoded[0]})

if __name__ == '__main__':
    app.run(debug=True, port=8080)

