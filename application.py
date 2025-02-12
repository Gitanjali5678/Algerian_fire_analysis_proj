import pickle 
from flask import Flask, request , jsonify , render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
app=application

##ridgeregression and Satandard scalar
ridge_model=pickle.load(open('model/ridge.pkl','rb'))
standard_model=pickle.load(open('model/scaler.pkl','rb'))

@app.route("/")
def index():
        return render_template('index.html')
        
    ##return render_template('index.html')

@app.route("/predictdata", methods=['POST', 'GET'])
def predict_data():
    try:
        if request.method == 'POST':
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            WS = float(request.form['WS'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Classes = float(request.form['Classes'])
            Region = float(request.form['Region'])

            # Prepare the input features as an array (for scaling and prediction)
            features = np.array([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
            print(features)
            # Scale features using your pre-trained scaler
            scaled_features = standard_model.transform(features)

            # Predict the result using your pre-trained model
            prediction = ridge_model.predict(scaled_features)
            print(f"Prediction: {prediction[0]}")
            # Render the template and pass the prediction result
            return render_template('home.html', prediction=prediction[0])
        else:
            return render_template('home.html')
    except Exception as e:
        # Log the error message and show it to the user
        print(f"Error occurred: {e}")
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5001)


