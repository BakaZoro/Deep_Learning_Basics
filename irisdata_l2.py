
###USING L2 REGULARISATION

import pandas as pd
import numpy as np


#READING DATA
data = pd.read_csv('iris.data', header=None)
data_X = data.iloc[:,:4].values
data_y = data.iloc[:,4].values
# print(data_X.shape)

#LABEL ENCODING STRINGS TO NUMBERS
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(data_y) #encoding the string to give a class label

#NORMALISING DATA: Z-SCORE NORMALISATION

mean=[]
mean=data_X.mean(0)

std_dev=[]
std_dev=data_X.std(0)

for i in range(data_X.shape[1]):
	for j in range(data_X.shape[0]):
		data_X[j,i]=(data_X[j,i]-mean[i])/std_dev[i]

iris_data=np.column_stack((data_X,y_enc)) #merging the two tables
# print(iris_data)


outerloop=10
innerloop=10

from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

f = open("IRIS-DATA_OUTPUT_L2.txt", "a")

outer=0
mean_acc3=0 #mean accuracy of outermost loop



for i in range(outerloop): #No of times the entire model is run

	#K-FOLDS 
	kf = KFold(n_splits=10,shuffle=True, random_state=None)
	kf.get_n_splits(iris_data)
	c=0 #count of weight iteration
	mean_acc2=0 #mean accuracy of second loop

	#INDICES ARE ALLOTED 
	for train_index, test_index in kf.split(iris_data):
		iris_data_train, iris_data_test = iris_data[train_index], iris_data[test_index]
		

		#TRAIN DATA
		iris_data_train_features = iris_data_train[:,:4]
		iris_data_train_label = iris_data_train[:,4]

		#TEST DATA
		iris_data_test_features = iris_data_test[:,:4]
		iris_data_test_label = iris_data_test[:,4]

		from keras.utils import to_categorical
		iris_data_train_label=to_categorical(iris_data_train_label)
		iris_data_test_label=to_categorical(iris_data_test_label)
		# print(iris_data_train_label)
		# print(iris_data_train_label.shape)
		
		mean_acc1=0 #mean accuracy calculated from inner loop

		

		for j in range(innerloop): #No of times the weight is initialised for each train and test dataset
			
			#CREATING NEURAL NETWORK
			inp=iris_data_train_features.shape[1]
			model = Sequential()
			model.add(Dense(5, input_dim=inp,kernel_regularizer=regularizers.l2(0.01),activation='sigmoid',use_bias=True,kernel_initializer='glorot_uniform')) 
			model.add(Dense(3, kernel_regularizer=regularizers.l2(0.01),activation='softmax',use_bias=True,kernel_initializer='glorot_uniform'))#Hidden and output layer
			# print(model.summary())

			#COMPILING THE MODEL
			from keras import optimizers
			adagrad=optimizers.Adagrad(learning_rate=0.1)
			model.compile(optimizer=adagrad,loss='mean_squared_error',metrics=['accuracy'])

			# TRAIN AND TEST THE MODEL
			history = model.fit(iris_data_train_features, iris_data_train_label,batch_size=15,epochs=1000,verbose=1,validation_data=(iris_data_test_features, iris_data_test_label))
			score = model.evaluate(iris_data_test_features, iris_data_test_label, verbose=0)
			# print('Test loss:', score[0])
			# print('Test accuracy:', score[1])
			mean_acc1+=score[1]
		c+=1
		mean_acc1/=innerloop
		f.write("For "+str(c)+" weight initialisation "+str(mean_acc1)+"\n")

		mean_acc2+=mean_acc1
		
	mean_acc2/=10
	f.write("For 10 folds: "+str(mean_acc2)+"\n")
	
	mean_acc3+=mean_acc2
outer+=1
mean_acc3/=outerloop
f.write("Outermost loop "+str(outer)+": "+str(mean_acc3)+"\n")
f.write("---------------------------------------------------------------------\n")


