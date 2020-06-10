from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import os
import pickle

MODEL_DIR = os.getcwd() + '/models/'
MODEL_NAME_SVM = 'full_data_SVM_model.pkl'
MODEL_NAME_XGB = 'full_data_XGB_model.pkl'
MODEL_NAME_BAG = 'full_data_BAG_model.pkl'
MODEL_NAME_RF = 'full_data_RF_model.pkl'
MODEL_NAME_KNN = 'full_data_KNN_model.pkl'

model_svm = pickle.load(open(MODEL_DIR + MODEL_NAME_SVM, 'rb'))
model_xgb = pickle.load(open(MODEL_DIR + MODEL_NAME_XGB, 'rb'))
model_bag = pickle.load(open(MODEL_DIR + MODEL_NAME_BAG, 'rb'))
model_rf = pickle.load(open(MODEL_DIR + MODEL_NAME_RF, 'rb'))
model_knn = pickle.load(open(MODEL_DIR + MODEL_NAME_KNN, 'rb'))

def predict_digit(data):
    result_svm = model_svm.predict(data)
    result_xgb = model_xgb.predict(data)
    result_bag = model_bag.predict(data)
    result_rf = model_rf.predict(data)
    result_knn = model_knn.predict(data)
    return [result_svm[0], result_xgb[0], result_bag[0], result_rf[0], result_knn[0]]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = (255 - np.array(request.json, dtype=np.uint8)).reshape(1, 784)
    np.save('data.npy', data)
    result = predict_digit(data)
    return jsonify(data=result)

if __name__ == '__main__':
    app.run(debug=True)