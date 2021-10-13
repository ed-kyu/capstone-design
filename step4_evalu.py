# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:07:44 2019

@author: NSMLAB
"""
import os
#os.environ["CUDA_VISIBLE_DEVICE"] = "2"
import numpy as np
from keras.models import *
import binvox_rw



def predict():
    print("-"*30)
    print("do the predict")
    print("-"*30)
    
    model = load_model("../../model/run1_70cases.h5")
    
    data = np.load("../../npy/fortest26_.npz")
#    X_test, Y_test = shuffle(data['X_test'], data['Y_test'])
    X_test, Y_test = data['X_test'], data['Y_test']

    print(X_test.shape)
    print(Y_test.shape)
#    X_test = X_test / 255
    Y_pred = model.predict(X_test)
#    Y_pred = model.test_on_batch(X_test, Y_test)

    np.save("../../npy/predict/run1_70cases.npy", Y_pred)

num = str(11)

def npy2binvox():
    print("-"*30)
    print(".npy to .binvox")
    print("-"*30)    
      
    Y_pred = np.load("../../npy/predict/run1_70cases.npy")
    Y_testname = np.load("../../npy/fortestname26_.npy")
    print(Y_pred.shape)
    

    for n in range(Y_pred.shape[0]):
        name = Y_testname[n]
        dims = Y_pred[n].shape[1] - 6
        print(name)

        Y_pred[Y_pred>0.5]=True
        Y_pred[Y_pred<=0.5]=False
        
        # To read binvox file, open file with read binary mode
        with open('../../result/run' + num + '/' + name,'rb+') as file:

            model = binvox_rw.read_as_3d_array(file)

        # predicted data to temp binvox file
            for i in range(dims):
                for j in range(dims):
                    for k in range(dims):
                        model.data[i][j][k] = bool(Y_pred[n,i+3,j+3,k+3,0])

        # To save the model, again open file with write binary mode
        with open('../../result/run' + num + '/pred_' + name,'wb') as file:
            model.write(file)

def trainbatch():
    print("-"*30)
    print("train on batch")
    print("-"*30)      
    
    x_batch = np.load('../../npy/forbatch26_x.npy')
    y_batch = np.load('../../npy/forbatch26_y.npy')
    
    model = load_model("../../model/run3_46cases.h5")
    
    print('train on batch')
    model.train_on_batch(x_batch, y_batch)
    print('save the model')
    model.save("../../model/run3_48cases.h5")

if __name__ == "__main__":
    
    predict()
    npy2binvox()
    
#    trainbatch()