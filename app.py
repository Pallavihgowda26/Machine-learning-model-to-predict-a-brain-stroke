from flask import Flask,render_template,url_for,request

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 


import pickle

app = Flask(__name__)

random = pickle.load(open('brain_random.pkl','rb'))
bagging = pickle.load(open('brain_bagging.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        age = request.form['age']
        hypertension = request.form['hypertension']
        heart_disease = request.form['heart_disease']
        ever_married = request.form['ever_married'] 
        work_type = request.form['work_type']
        residence_type = request.form['residence_type']
        avg_glucose_level = request.form['avg_glucose_level']
        bmi = request.form['bmi']
        smoking_status = request.form['smoking_status']
        
        
        model = request.form['model']
        
		# Clean the data by convert from unicode to float 
        
        sample_data = [gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]
        # clean_data = [float(i) for i in sample_data]
        # int_feature = [x for x in sample_data]
        int_feature = [float(i) for i in sample_data]
        print(int_feature)
    

		# Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(int_feature).reshape(1,-1)
        print(ex1)
		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if model == 'RandomForestClassifier':
           result_prediction = random.predict(ex1)
           
            
        elif model == 'BaggingClassifier':
          result_prediction = bagging.predict(ex1)
           
           
        
        if result_prediction > 0.5:
            result = 'Stroke'
        else:
            result = 'No Stroke'    

          

    return render_template('prediction.html', prediction_text= result, model = model)

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")    

if __name__ == '__main__':
	app.run(debug=True)
