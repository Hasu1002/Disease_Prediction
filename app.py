from flask import Flask, render_template, request
from implementation import randorm_forest_test, random_forest_train, random_forest_predict
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random_forest import accuracy
from sklearn.metrics import accuracy_score
from time import time
import pickle


app = Flask(__name__)
app.url_map.strict_slashes = False
filename = 'heart.pkl'
heartmodel = pickle.load(open(filename, 'rb'))

@app.route('/')
def homepage():
	return render_template('homepage.html')

@app.route('/bcancerhome')
def index():
	return render_template('bcancerhome.html')

@app.route('/bcancerpredict', methods=['POST']) 
def login_user():

	data_points = list()
	data = []
	string = 'value'
	for i in range(1,31):
		data.append(float(request.form['value'+str(i)]))

	for i in range(30):
		data_points.append(data[i])
		
	print(data_points)

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
	out, acc, t = random_forest_predict(clf, data_np)

	if(out==1):
		output = 'Malignant'
	else:
		output = 'Benign'

	acc_x = acc[0][0]
	acc_y = acc[0][1]
	if(acc_x>acc_y):
		acc1 = acc_x
	else:
		acc1=acc_y
	return render_template('bcancerresult.html', output=output, accuracy=accuracy, time=t)

kidneymodel = pickle.load(open('Kidneyn.pkl', 'rb'))

@app.route('/heart', methods=["POST", "GET"])
def heart():
    if request.method == 'POST':
        myDict3 = request.form
        age = int(myDict3['age'])
        sex = int(myDict3['sex'])
        cp = int(myDict3['cp'])
        trestbps = int(myDict3['trestbps'])
        chol = int(myDict3['chol'])
        fbs = float(myDict3['fbs'])
        restecg = float(myDict3['restecg'])
        thalach = int(myDict3['thalach'])
        exang = int(myDict3['exang'])
        oldpeak = int(myDict3['oldpeak'])
        slope = int(myDict3['slope'])
        ca = int(myDict3['ca'])
        thal = int(myDict3['thal'])

        data1 = np.array([[age, sex, cp, trestbps, chol, fbs,
                           restecg, thalach, exang, oldpeak, slope, ca, thal]])
        my_prediction1 = heartmodel.predict(data1)

        return render_template('heartshow.html', prediction1=my_prediction1)
    return render_template('heart.html')

@app.route('/kidney',methods=['GET'])
def Home():
    return render_template('kidneyindex.html')

@app.route("/kidneypredict", methods=['POST'])
def kidneypredict():
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction = kidneymodel.predict(values)

        return render_template('kidneyresult.html', prediction=prediction)
	
diascaler = pickle.load(open('scaler.pkl', 'rb'))
diamodel = pickle.load(open('svm_model.pkl', 'rb'))

@app.route('/dia', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        pregs = int(request.form.get('pregs'))
        gluc = int(request.form.get('gluc'))
        bp = int(request.form.get('bp'))
        skin = int(request.form.get('skin'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        func = float(request.form.get('func'))
        age = int(request.form.get('age'))

        input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
        # print(input_features)
        prediction = diamodel.predict(diascaler.transform(input_features))
        # print(prediction)
        
    return render_template('diaindex.html', prediction=prediction)

	

if __name__=='__main__':
	global clf 
	clf = random_forest_train()
	randorm_forest_test(clf)
	#print("Done")
	app.run(debug=True)

