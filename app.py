from flask import Flask, redirect, url_for, request,render_template,jsonify
import numpy as np
import base64
from PIL import Image
from keras.models import load_model
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')



from io import BytesIO
import re, time, base64



@app.route('/predict',methods=['POST'])
def predict():
    if request.method=="POST":
        data = request.form['data']
        print('doing')
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
        print(proper_img.shape)
        out = model.predict_classes(proper_img)
        return str(out[0])
    

if __name__ == '__main__':
    model = load_model('longer.h5')
    model._make_predict_function()
    port = int(os.environ.get('PORT',5000))
    app.run()
