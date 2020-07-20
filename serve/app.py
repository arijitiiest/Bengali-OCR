from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import os
import pickle
import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session


# from tensorflow import keras
# ### hack tf-keras to appear as top level keras
# import sys
# sys.modules['keras'] = keras


MODEL_DIR = os.getcwd() + '/models/'
MODEL_LENET_JSON = 'LeNet_model.json'
MODEL_RESNET_JSON = 'ResNet_model.json'
MODEL_LENET_WEIGHTS = 'LeNet_model.h5'
MODEL_RESNET_WEIGHTS = 'ResNet_model.h5'

lenet_json_file = open(MODEL_DIR + MODEL_LENET_JSON)
lenet_loaded_model_json = lenet_json_file.read()
lenet_json_file.close()

resnet_json_file = open(MODEL_DIR + MODEL_RESNET_JSON)
resnet_loaded_model_json = resnet_json_file.read()
resnet_json_file.close()

model_LeNet = model_from_json(lenet_loaded_model_json)
model_LeNet.load_weights(MODEL_DIR + MODEL_LENET_WEIGHTS)
# lenet_adam = Adam(lr=5e-4)
# model_LeNet.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=lenet_adam)
# model_LeNet._make_predict_function()

model_ResNet = model_from_json(resnet_loaded_model_json)
model_ResNet.load_weights(MODEL_DIR + MODEL_RESNET_WEIGHTS)
# resnet_adam = Adam(lr=0.0001)
# model_ResNet.compile(optimizer= resnet_adam, loss='categorical_crossentropy', metrics=['accuracy'])
# model_ResNet._make_predict_function()


tf_config = some_custom_config
sess = tf.Session(graph=tf.Graph())
# graph = tf.get_default_graph()


def predict_digit(data):
    global sess
    global graph
    # with graph.as_default():
        # set_session(sess)
    lenet_predict = model_LeNet.predict(data)
    resnet_predict = model_ResNet.predict(data)
    print(lenet_predict)
    print(resnet_predict)
    return [lenet_predict[0], resnet_predict[0]]
    # return []


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