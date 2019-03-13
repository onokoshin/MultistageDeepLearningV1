from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import date
from werkzeug.utils import secure_filename
import numpy as np
import os
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import DamageDetectionFlask.single_image_model as mask_rcnn



basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'templates\img')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



app._static_folder = os.path.join(basedir, "templates")

model_name = 'bumper_damage_front_rear_021019_224pix_vgg19_00001.h5'

basedir = os.path.abspath(os.path.dirname(__file__))

model_path = os.path.join(basedir, model_name)

model = None
graph = None

@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        r = request

        f = r.files['image']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(img_path)

        # pred_result = pred.get_prediction(img_path)

        img = image.load_img(path=img_path, grayscale=False, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        pred_result = ''
        with graph.as_default():
            pred_result = model.predict(x)

        static_img_path = os.path.join('static\img', f.filename)
        # print(img_path)
        # print(static_img_path)
        # print(pred_result)

        mask_image_path = mask_rcnn.get_image(img_path, f.filename)
        # print(mask_image_path)
        ls = mask_image_path.split('/')
        static_mask_img_path = os.path.join('static', ls[-1])

        # os.remove(f.filename)
        # return redirect(url_for('asessment', image=static_img_path, prediction=pred))

        # flip it to percentage
        yes_pred = str(round(pred_result[0][1] * 100, 2)) + '%'
        no_pred = str(round(pred_result[0][0] * 100, 2)) + '%'


        return render_template('asessment.html',
                               image_name=f.filename,
                               image=static_img_path,
                               mask_image=static_mask_img_path,
                               pred_yes=yes_pred,
                               pred_no=no_pred)



@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/asessment')
def asessment():

    return render_template('asessment.html', pred_yes='', pred_no='')

@app.route('/index')
def index():
    yr = date.today().year
    return render_template('home.html', year=yr)


def load_h5_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model, graph
    model_name = 'bumper_damage_front_rear_021019_224pix_vgg19_00001.h5'
    basedir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(basedir, model_name)
    model = load_model(filepath=model_path)
    graph = tf.get_default_graph()


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_h5_model()
    mask_rcnn.load_mask_rcnn_model()
    app.run(debug=True)
