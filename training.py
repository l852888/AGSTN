import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend
 
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
#======read data===================================================
Y=np.load(r"Total_Y.npy")
graph_conv_filter=preprocess_adj_tensor_with_identity(matrix) #matrix:edge weight of graph.
                                                              #we utilize cosine similarity to bulid the relationship between each sensors.
X_train=final[0:train_size]
X_test=final[train_size:]
X_train1=final1[0:train_size]
X_test1=final1[train_size:]
X_train2=final2[0:train_size]
X_test2=final2[train_size:]
X_train3=final3[0:train_size]
X_test3=final3[train_size:]
X_train4=final4[0:train_size]
X_test4=final4[train_size:]
X_train5=final5[0:train_size]
X_test5=final5[train_size:]
y_train=Y[0:train_size]
y_test=Y[train_size:]
cnnX_train=finalcnn[0:train_size]
cnnX_test=finalcnn[train_size:]
MX_train=graph_conv_filter[0:train_size]
MX_test=graph_conv_filter[train_size:]
tX_train=finalcnn[0:train_size]
tX_test=finalcnn[train_size:]

#===========train model================================
import keras
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Lambda,Flatten,LSTM,AveragePooling1D,GRU,Conv2D,AveragePooling2D,GRU,Multiply
import keras.backend as K
from sklearn.utils import shuffle

past_steps=6
number_of_sensors=26
num_filters = 2
graph_conv_filters_input = Input(shape=(number_of_sensors*2, number_of_sensors))
    
tinput=Input(shape=(past_steps,number_of_sensors))
t=timeattention(number_of_sensors)(tinput)
t=Flatten()(t)

#X_input1 to X_input6 means that there are 6 past_steps
X_input1 = Input(shape=(number_of_sensors, 1))
X_input2 = Input(shape=(number_of_sensors, 1))
X_input3 = Input(shape=(number_of_sensors, 1))
X_input4 = Input(shape=(number_of_sensors, 1))
X_input5 = Input(shape=(number_of_sensors, 1))
X_input6 = Input(shape=(number_of_sensors, 1))

output1 = MultiGraphCNN(1, num_filters)([X_input1, graph_conv_filters_input])
output1 = MultiGraphCNN(1, num_filters)([output1, graph_conv_filters_input])
output1 = MultiGraphCNN(1, num_filters)([output1, graph_conv_filters_input])
output1=Flatten()(output1) 

output2 = MultiGraphCNN(1, num_filters)([X_input2, graph_conv_filters_input])
output2 = MultiGraphCNN(1, num_filters)([output2, graph_conv_filters_input])
output2 = MultiGraphCNN(1, num_filters)([output2, graph_conv_filters_input])
output2=Flatten()(output2)
  
output3 = MultiGraphCNN(1, num_filters)([X_input3, graph_conv_filters_input])
output3 = MultiGraphCNN(1, num_filters)([output3, graph_conv_filters_input])
output3 = MultiGraphCNN(1, num_filters)([output3, graph_conv_filters_input])
output3=Flatten()(output3)

output4 = MultiGraphCNN(1, num_filters)([X_input4, graph_conv_filters_input])
output4 = MultiGraphCNN(1, num_filters)([output4, graph_conv_filters_input])
output4 = MultiGraphCNN(1, num_filters)([output4, graph_conv_filters_input])
output4=Flatten()(output4)

output5 = MultiGraphCNN(1, num_filters)([X_input5, graph_conv_filters_input])
output5 = MultiGraphCNN(1, num_filters)([output5, graph_conv_filters_input])
output5 = MultiGraphCNN(1, num_filters)([output5, graph_conv_filters_input])
output5=Flatten()(output5)

output6 = MultiGraphCNN(1, num_filters)([X_input6, graph_conv_filters_input])
output6 = MultiGraphCNN(1, num_filters)([output6, graph_conv_filters_input])
output6 = MultiGraphCNN(1, num_filters)([output6, graph_conv_filters_input])
output6=Flatten()(output6)

con=keras.layers.concatenate([output1,output2,output3,output4,output5,output6])
con2=keras.layers.Reshape((past_steps,number_of_sensors))(con)
con2=keras.layers.Reshape((past_steps,number_of_sensors,1))(con2)
#cnn
cnnoutput=Conv2D(1,3,1)(con2)
maxpooling=AveragePooling2D((4,1))(cnnoutput)
cnnoutput=Flatten()(maxpooling)
cnninput=Input(shape=(past_steps,number_of_sensors))

#LSTM
output=LSTM(number_of_sensors)(cnninput)
output1=Dense(number_of_sensors)(output)
 
#Learning attention weight for adjustment
output=keras.layers.average([output1,cnnoutput])
output=Multiply()([output,t])
    
nb_epochs = 50
batch_size = 32
   
model1 = Model(inputs=[X_input1,X_input2,X_input3,X_input4,X_input5,X_input6,graph_conv_filters_input,cnninput,tinput], outputs=output)
model1.summary()
Adam =keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model1.compile(loss='mse', optimizer=Adam, metrics=['mae',rmse])
history=model1.fit([np.array(X_train),np.array(X_train1),np.array(X_train2)
           ,np.array(X_train3),np.array(X_train4),np.array(X_train5),np.array(MX_train),np.array(cnnX_train),np.array(tX_train)],np.array(y_train),
          batch_size=batch_size, validation_split=0.1, 
          epochs=nb_epochs, shuffle=True, verbose=1)
score=model1.evaluate([np.array(X_test),np.array(X_test1),np.array(X_test2),np.array(X_test3)
                ,np.array(X_test4),np.array(X_test5),np.array(MX_test),np.array(cnnX_test),np.array(tX_test)],np.array(y_test))
predict1=model1.predict([np.array(X_test),np.array(X_test1),np.array(X_test2),np.array(X_test3)
                ,np.array(X_test4),np.array(X_test5),np.array(MX_test),np.array(cnnX_test),np.array(tX_test)])
    
