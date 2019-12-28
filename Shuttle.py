
#We want to obtain a video showing how the predicted values come closer to the actual values with time over the enitre back propagation process on the IRIS dataset.

import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

#READING DATA
data = pd.read_csv('iris.data', header=None)
data_X = data.iloc[:,:4].values
data_y = data.iloc[:,4].values

#LABEL ENCODING THE CLASS LABEL
le = LabelEncoder()
y_enc = le.fit_transform(data_y)

#NORMALISING THE FEATURES
mean=[]
mean=data_X.mean(0)

std_dev=[]
std_dev=data_X.std(0)

for i in range(data_X.shape[1]):
	for j in range(data_X.shape[0]):
		data_X[j,i]=(data_X[j,i]-mean[i])/std_dev[i]  ##Z SCORE


#MERGING THE TWO MATRIXES
iris_data=np.column_stack((data_X,y_enc))

#CREATING NEURAL NETWORK

model = Sequential()
model.add(Dense(4, input_shape=(4,),activation='linear',use_bias=True,kernel_initializer='glorot_uniform')) 
model.add(Dense(5,activation='relu',use_bias=True,kernel_initializer='glorot_uniform')) 
model.add(Dense(3,activation='sigmoid',use_bias=True,kernel_initializer='glorot_uniform'))


#ONE HOT ENCODING THE FEATURES
y_enc=to_categorical(y_enc)

#COMPILING THE MODEL

model.compile(optimizer='adagrad',loss='categorical_crossentropy',metrics=['accuracy'])

#CUSTOM CALLBACK

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

class PredictionCallback(tf.keras.callbacks.Callback): 
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(data_X)
    ax.scatter(y_pred[:,0],y_pred[:,1],y_pred[:,2],c='b')
    ax.scatter(y_enc[:,0],y_enc[:,1],y_enc[:,2],c='r')
    # ax.plot(y_pred,y_enc,c='g')
    ax.set_xlabel("Class 0")
    ax.set_ylabel("Class 1")
    ax.set_zlabel("Class 2")
    plt.savefig(str(epoch)+".png")
    plt.cla()



#TRAINING THE MODEL

model.fit(data_X, y_enc, batch_size=15, epochs=1000,callbacks=[PredictionCallback()],verbose=1)
# plt.show()
