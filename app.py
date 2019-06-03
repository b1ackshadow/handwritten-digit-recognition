from flask import Flask, redirect, url_for, request,render_template,jsonify
import numpy as np
import base64
from PIL import Image
from keras.models import load_model
import os
import sys
import logging
import matplotlib.pyplot as plt
import json
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

model = load_model('longer.h5')
model._make_predict_function()

@app.route('/')
def index():
    return render_template('index.html')



from io import BytesIO
import re, time, base64



@app.route('/predict',methods=['POST'])
def predict():
    if request.method=="POST":
        data = request.form['data']
        # print('doing')
        img = base64.b64decode(data)
        # print(img)
        fname = 'input.png'
        with open(fname,'wb') as f:
            f.write(img)
            f.close()
        

        img = Image.open(fname).convert('LA')
        basewidth = 28

        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save(fname) 
        img = Image.open(fname).convert('LA')
        proper_img = (np.array(img)[:,:,0]).reshape(1,28,28,1)
        # print(proper_img.shape)
        data = {}
        proba = model.predict_proba(proper_img).ravel().astype(np.float).tolist()
        data['out'] = int(np.argmax(proba))
        print(data,proba)
        plt.bar(list(range(10)),proba)
        plt.xticks(list(range(10)))
        plt.savefig('proba_dist.png')
        # plt.show()
        with open("proba_dist.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        data['proba'] = encoded_string.decode('utf-8')
        # result = 'proba:""'
        return jsonify(data)
    

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(debug=True)
