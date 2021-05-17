from flask import Flask,render_template,url_for,request,jsonify, make_response
import base64
#from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle
import cv2
import os,io,sys
from PIL import Image,ImageDraw
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__, template_folder="template")

MODEL_PATH = 'model/model_fullimage.h5'
model = load_model(MODEL_PATH)

print("Model Loaded")

@app.route("/",methods=['GET'])
#@cross_origin()
def home():
	return render_template("index.html")


def model_predict(img_path, model):
	img1=Image.open(img_path)
	img = image.load_img(img_path, target_size=(64,64))

	# Preprocessing the image
	x = image.img_to_array(img)
	x = x/255
	x = np.expand_dims(x, axis=0)

	preds = model.predict(x)
	return preds




@app.route('/predict', methods=['GET','POST'])
def upload():
    print("hello")
    if(request.method == 'POST'):
        # Get the file from post request
        print("hello2")
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
       # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        preds=preds[0]*1935
        x = []
        y = []
        j = 1
        for i in preds:
            if j % 2 != 0:
                x.append(i)
            else:
                y.append(i)
            j = j + 1
        img = Image.open(file_path)
        draw = ImageDraw.Draw(img)
        for t,q in zip(x,y):
            draw.ellipse((t - 8, q - 8, t + 8, q + 8), fill=(255, 0, 0, 0))
        p = 'uploads/pic' + '.png'
        img.save(p, "JPEG")
        with open("uploads/pic.png", "rb") as image_file:
            image_file=image_file.read()
            npimg = np.frombuffer(image_file, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            img = Image.fromarray(img.astype("uint8"))
            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.read())
        return img_base64
    return render_template("predictor.html")



if __name__=='__main__':
	app.run(debug=True)
