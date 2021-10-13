from keras.preprocessing.image import ImageDataGenerator
import os 
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, Dropout, Cropping3D, concatenate, Activation, Conv3DTranspose
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from keras import backend as keras
#from data import *
from keras.utils import multi_gpu_model
#from metrics import *
from keras.losses import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



batch_size = 4
epochs = 300
resol = 32

class myUnet(object):
	def __init__(self, img_size = resol, img_depth = resol):
		self.img_size = img_size
		self.img_depth = img_depth

	def get_unet(self):
		inputs = Input((self.img_size, self.img_size, self.img_depth,1))

#		inputs = Reshape((self.img_size, self.img_size, self.img_depth, 1), input_shape=(self.img_size, self.img_size, self.img_depth))
		conv1 = Conv3D(32, (3, 3, 3), activation=None, padding='same')(inputs)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		conv1 = Conv3D(32, (3, 3, 3), activation=None, padding='same')(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

		conv2 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(pool1)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		conv2 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(conv2)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

		conv3 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(pool2)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		conv3 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(conv3)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

		conv4 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(pool3)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		conv4 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(conv4)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

		conv5 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(pool4)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)
		conv5 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(conv5)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)
		
		up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4])
		conv6 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(up6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)
		conv6 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(conv6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)
		
		up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3])
#		up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(pool3), conv3])
		conv7 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(up7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)
		conv7 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(conv7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)
		
		up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2])
		conv8 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(up8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)
		conv8 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(conv8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)
		
		up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1])
		conv9 = Conv3D(32, (3, 3, 3), activation=None, padding='same')(up9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation('relu')(conv9)
		conv9 = Conv3D(32, (3, 3, 3), activation=None, padding='same')(conv9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation('relu')(conv9)
		

		conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

		model = Model(inputs=inputs, outputs=conv10)        
		model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999), loss='binary_crossentropy', metrics=['accuracy'])
#		model.summary()


		return model


	def train(self):
		data = np.load("../../npy/forlearning26_.npz")
#		print(data)
		X, Y = shuffle(data['X_train'], data['Y_train'])
		print(X.shape)
		print(Y.shape)
		data_test = np.load("../../npy/fortest26_.npz")
		X_test, Y_test = shuffle(data_test['X_test'], data_test['Y_test'])        

#		X = X/255
#		Y = Y/255
#		Y[Y>0.5]=1
#		Y[Y<=0.5]=0


		model = self.get_unet()
		print("got unet")
		model=multi_gpu_model(model,gpus=2)
#		model.summary()
		model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999), loss='binary_crossentropy', metrics=['accuracy'])
        
		# fit the generators
		print('Fitting model...')
		hist=model.fit(X, Y, batch_size=batch_size, epochs = epochs, validation_split=0.2, shuffle=True)
#		hist=model.fit(X, Y, batch_size=batch_size, epochs = epochs, shuffle=True)

		model.save("../../model/run1_70cases.h5")
		
	
		#make plot
		       
		plt.plot(hist.history['loss'], 'y', lw=3, ls="--", marker=".", ms=3, mec='y', label='train loss')
		plt.plot(hist.history['val_loss'], 'r', lw=3, ls="-", marker="o", ms=3, mec='r', label='val loss')
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.legend(loc='upper left')
		plt.grid(False)
#		fig1 = plt.gcf()
		plt.show()
#		fig1.savefig('plot_loss4_26.png')
 
		plt.plot(hist.history['accuracy'], 'y', lw=3, ls="--", marker=".", ms=3, mec='y', label='train accuracy')
		plt.plot(hist.history['val_accuracy'], 'r', lw=3, ls="-", marker="o", ms=3, mec='r', label='val accuracy')
		plt.xlabel('epoch')
		plt.ylabel('accuracy')
		plt.legend(loc='upper left')
		plt.grid(False)
#		fig2 = plt.gcf()
		plt.show()
#		fig2.savefig('plot_accuracy4_26.png')

		
		
	   
if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
