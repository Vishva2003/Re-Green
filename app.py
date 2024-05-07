# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
from markupsafe import Markup
from disease_dic import disease_dic
from fertilizer_dic import fertilizer_dic
import requests
import json
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from model import ResNet9
from flask import jsonify
from flask import Flask, render_template, request, redirect, session
import mysql.connector
import re
import os 


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS ---------------  #85c3cd  --------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
disease_model_path = 'Trained_Model/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = '.\Trained_Model\RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)


app.secret_key=os.urandom(24)
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Vishva2003',
    database='portfolio'
)
cursor = conn.cursor()

#Routing to the user Login page

@app.route('/')
def login():
   return render_template("login.html")



@app.route("/login1", methods=['GET', 'POST'])
def login1():
    msglog =""
    USERNAME = request.form['username']
    PASSWORD = request.form['password']
    checked = request.form.get('check')
    if checked:
        if request.method == 'POST':
            USERNAME = request.form['username']
            PASSWORD = request.form['password']

            query = "SELECT * FROM user WHERE USERNAME = %s AND PASSWORD = %s"
            cursor.execute(query, (USERNAME, PASSWORD))
            user=cursor.fetchall()

            if len(user)>0:
                session["USER_ID"]=user[0][0]
                return redirect("/home")
            else:
                error="Invalid username or password"
                return render_template("login.html", msglog=error)

        return render_template('login.html')
    else:
        error="Please check 'Remember me' to stay logged in."
        return render_template("login.html", msglog=error)

#Routing to the user registration page


@app.route("/reg1", methods=['GET', 'POST'])
def reg1():
    msgreg=""
    if request.method == 'POST':
        USERNAME = request.form['username']
        PASSWORD = request.form['password']
        EMAIL = request.form['email']

        # Check if account exists using MySQL
        query = "SELECT * FROM user WHERE USERNAME = %s AND PASSWORD = %s"
        cursor.execute(query, (USERNAME, PASSWORD))
        user = cursor.fetchall()
        # If account exists show error and validation checks
        if user:
            msg = 'Account already exists!'
            return render_template('login.html', msglog=msg)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', EMAIL):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', USERNAME):
            msg = 'Username must contain only characters and numbers!'
        elif not USERNAME or not PASSWORD or not EMAIL:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            query =("INSERT INTO user(USER_ID,USERNAME ,PASSWORD ,EMAIL ) VALUES (NULL, %s, %s, %s)")  
            cursor.execute(query, (USERNAME, PASSWORD,EMAIL,))
            user=conn.commit()
            msg = 'You have successfully registered!'
            return render_template('login.html', msglog=msg)
    else :
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('login.html', msgreg=msg)

@ app.route('/home')
def home():
    return render_template('index.html')

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html')

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html')

# render disease prediction input page


@ app.route('/disease')
def disease():
    return render_template('disease.html')

# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction)

        else:

            return render_template('try_again.html')

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
   
    
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/FertilizerData.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img = file.read()

        prediction = predict_image(img)

        prediction = Markup(str(disease_dic[prediction]))
        return render_template('disease-result.html', prediction=prediction)

    return render_template('disease.html')

@app.route('/test')
def test():
    message = {"Name": "Isariopsis_Leaf_Spot", "Type": "Leaf Spot"}
    return message

# ===============================================================================================

# ===============================================================================================
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
