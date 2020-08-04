from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import os
import pickle
import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session



MODEL_DIR = os.getcwd() + '/models/'
MODEL_LENET_JSON = 'LeNet_model.json'
MODEL_RESNET_JSON = 'ResNet_model.json'
MODEL_LENET_WEIGHTS = 'LeNet_model.h5'
MODEL_RESNET_WEIGHTS = 'ResNet_model.h5'



# Load LeNet Model
sess1 = tf.Session()
graph1 = tf.compat.v1.get_default_graph()

set_session(sess1)
lenet_json_file = open(MODEL_DIR + MODEL_LENET_JSON)
lenet_loaded_model_json = lenet_json_file.read()
lenet_json_file.close()

model_LeNet = model_from_json(lenet_loaded_model_json)
model_LeNet.load_weights(MODEL_DIR + MODEL_LENET_WEIGHTS)
lenet_adam = Adam(lr=5e-4)
model_LeNet.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=lenet_adam)

# Load ResNet Model
sess2 = tf.Session()
graph2 = tf.compat.v1.get_default_graph()

set_session(sess2)
resnet_json_file = open(MODEL_DIR + MODEL_RESNET_JSON)
resnet_loaded_model_json = resnet_json_file.read()
resnet_json_file.close()

model_ResNet = model_from_json(resnet_loaded_model_json)
model_ResNet.load_weights(MODEL_DIR + MODEL_RESNET_WEIGHTS)
resnet_adam = Adam(lr=0.0001)
model_ResNet.compile(optimizer= resnet_adam, loss='categorical_crossentropy', metrics=['accuracy'])


def predict_digit(data):
    global sess1
    global sess2
    global graph1
    global graph2

    with graph1.as_default():
        set_session(sess1)
        lenet_predict = model_LeNet.predict(data)
    
    with graph2.as_default():
        set_session(sess2)
        resnet_predict = model_ResNet.predict(data)

    lenet_value = np.where(lenet_predict[0] == np.amax(lenet_predict[0]))[0][0]
    resnet_value = np.where(resnet_predict[0] == np.amax(resnet_predict[0]))[0][0]

    return [str(lenet_value), str(resnet_value)]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = (255 - np.array(request.json, dtype=np.uint8)).reshape(1, 32, 32, 1)
    np.save('data.npy', data)
    result = predict_digit(data)
    return jsonify(data=result)

if __name__ == '__main__':
    app.run(debug=True)