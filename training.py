import pandas as pd
import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import train_test_split
    Y=np.load(r"C:\Users\User\Desktop\npy\Total_Y.npy")
    X_train=final[0:7003]
    X_test=final[7003:]
    X_train1=final1[0:7003]
    X_test1=final1[7003:]
    X_train2=final2[0:7003]
    X_test2=final2[7003:]
    X_train3=final3[0:7003]
    X_test3=final3[7003:]
    X_train4=final4[0:7003]
    X_test4=final4[7003:]
    X_train5=final5[0:7003]
    X_test5=final5[7003:]
    y_train=Y[0:7003]
    y_test=Y[7003:]
    cnnX_train=finalcnn[0:7003]
    cnnX_test=finalcnn[7003:]
    MX_train=graph_conv_filter[0:7003]
    MX_test=graph_conv_filter[7003:]
    tX_train=finalcnn[0:7003]
    tX_test=finalcnn[7003:]
    import keras
    from keras.models import Input, Model, Sequential
    from keras.layers import Dense, Activation, Dropout, Lambda,Flatten,LSTM,AveragePooling1D,GRU,Conv2D,AveragePooling2D,GRU,Multiply
    import keras.backend as K

    from sklearn.utils import shuffle

    num_filters = 2
    graph_conv_filters_input = Input(shape=(52, 26))
    
    tinput=Input(shape=(6,26))
    t=timeattention(26)(tinput)
    t=Flatten()(t)
    
    

    X_input1 = Input(shape=(26, 1))
    X_input2 = Input(shape=(26, 1))
    X_input3 = Input(shape=(26, 1))
    X_input4 = Input(shape=(26, 1))
    X_input5 = Input(shape=(26, 1))
    X_input6 = Input(shape=(26, 1))

    
    
    
    
    
    
    
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
    

    
    con2=keras.layers.Reshape((6,26))(con)
    
    con2=keras.layers.Reshape((6,26,1))(con2)
    cnnoutput=Conv2D(1,3,1)(con2)
    maxpooling=AveragePooling2D((4,1))(cnnoutput)
    cnnoutput=Flatten()(maxpooling)
    
    cnninput=Input(shape=(6,26))
  
    output=LSTM(26)(cnninput)
    output1=Dense(26)(output)
    
    
    

    
    
    output=keras.layers.average([output1,cnnoutput])
    
    output=Multiply()([output,t])
    







    nb_epochs = 50
    batch_size = 32
   
    model1 = Model(inputs=[X_input1,X_input2,X_input3,X_input4,X_input5,X_input6,graph_conv_filters_input,cnninput,tinput], outputs=output)
    model1.summary()
    RMSprop =keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model1.compile(loss='mse', optimizer=RMSprop, metrics=['mae',rmse])
    history=model1.fit([np.array(X_train),np.array(X_train1),np.array(X_train2)
           ,np.array(X_train3),np.array(X_train4),np.array(X_train5),np.array(MX_train),np.array(cnnX_train),np.array(tX_train)],np.array(y_train),
          batch_size=batch_size, validation_split=0.1, 
          epochs=nb_epochs, shuffle=True, verbose=1)
    score=model1.evaluate([np.array(X_test),np.array(X_test1),np.array(X_test2),np.array(X_test3)
                ,np.array(X_test4),np.array(X_test5),np.array(MX_test),np.array(cnnX_test),np.array(tX_test)],np.array(y_test))
    predict1=model1.predict([np.array(X_test),np.array(X_test1),np.array(X_test2),np.array(X_test3)
                ,np.array(X_test4),np.array(X_test5),np.array(MX_test),np.array(cnnX_test),np.array(tX_test)])
    
