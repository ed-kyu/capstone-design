# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:22:06 2019

@author: NSMLAB
"""

import os
import numpy as np
import binvox_rw
#import glob

def voxtonpz():
    print('-'*30)
    print('.binvox file to .npz file for learning')
    print('-'*30)
    
    X = {'train':[], 'test':[]}
    Y = {'train':[], 'test':[]}
    
    for split in ['train', 'test']:
#        x_path = os.path.join('../../data',split, 'x/')
        print(split)
        x_path = '../../data/' + split + '/x' # x is piston
        print(x_path)
        for x_dir in os.listdir(x_path):
            print(x_dir)
            with open(os.path.join(x_path,x_dir),'rb') as file:
                data=np.int32(binvox_rw.read_as_3d_array(file).data)
                data=np.pad(data,3,'constant')
#                print(data)
                X[split].append(data)         
                
        y_path = os.path.join('../../data',split, 'y/') # y is preform
        print(y_path)
        
        ytestname=[]
        for y_dir in os.listdir(y_path):
            print(y_dir)
            with open(os.path.join(y_path,y_dir),'rb') as file:
                data=np.int32(binvox_rw.read_as_3d_array(file).data)
                data=np.pad(data,3,'constant')

                Y[split].append(data)
                
            if split == 'test':
                ytestname.append(y_dir )
                
#    print(X.shape)
    X_train=np.array(X['train'])
    X_test=np.array(X['test'])
    Y_train=np.array(Y['train'])
    Y_test=np.array(Y['test'])

    X_train = X_train[:,:,:,:,np.newaxis]
    X_test = X_test[:,:,:,:,np.newaxis]
    Y_train = Y_train[:,:,:,:,np.newaxis]
    Y_test = Y_test[:,:,:,:,np.newaxis]
    print('X_train ', X_train.shape)
    print('X_test ', X_test.shape)
    print('Y_train ', Y_train.shape)
    print('Y_test ', Y_test.shape)    

    np.savez_compressed('../../npy/forlearning26_.npz',
                        X_train=X_train, Y_train=Y_train)
    np.savez_compressed('../../npy/fortest26_.npz',
                        X_test=X_test, Y_test=Y_test)    
    np.save('../../npy/fortestname26_.npy', ytestname)
    
def voxtonpy():
    print('-'*30)
    print('.binvox file to .npy file for batch learning')
    print('-'*30)

    x=[]; y=[]
    
    x_path = '../../data/batch/x'
    for x_dir in os.listdir(x_path):
        print(x_dir)
        with open(os.path.join(x_path,x_dir),'rb') as file:
            data=np.int32(binvox_rw.read_as_3d_array(file).data)
            data=np.pad(data,3,'constant')
            x.append(data)
    
    y_path = '../../data/batch/y'
    for y_dir in os.listdir(y_path):
        print(y_dir)
        with open(os.path.join(y_path,y_dir),'rb') as file:
            data=np.int32(binvox_rw.read_as_3d_array(file).data)
            data=np.pad(data,3,'constant')
            y.append(data)
    x = np.array(x); y = np.array(y)
    x_batch = x[:,:,:,:,np.newaxis]
    y_batch = y[:,:,:,:,np.newaxis]
    
    print('x_batch ', x_batch.shape)
    print('y_batch ', y_batch.shape)
    
    np.save('../../npy/forbatch26_x.npy', x_batch)
    np.save('../../npy/forbatch26_y.npy', y_batch)
    
if __name__ == "__main__":
    voxtonpz()
#    voxtonpy()