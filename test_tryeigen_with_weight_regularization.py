# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:56:49 2019

@author: Travis_Whyte
"""

from keras import backend as K
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.layers.core import Dense, Reshape, Dropout
from keras.layers import Input, Flatten
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt



class Params:
    def __init__(self):
        self.n = 16
        self.complexity = 1
        self.numtrain = 500
        
def return_params():
    return Params()

def get_diag_eigensystems(n,num):
    train_A = []
    train_eval = []
    
    for i in range(num):
        
        A = np.random.random_sample((n,n)).astype(np.float32)
        A = np.diag(np.diag(A))
        d,v = np.linalg.eig(A)
        
        train_A.append(A)
        train_eval.append(d)
        
    train_A = np.array(train_A)
    train_eval = np.array(train_eval)
    
    return train_A, train_eval

def get_tridiag_eigensystems(n,num):
    train_A = []
    train_eval = []
    
    for i in range(num):
        """
        A = np.random.random_sample((n,n)).astype(np.float32)
        b = np.random.random_sample((n,1)).astype(np.float32)
        c = np.random.random_sample((n,1)).astype(np.float32)
        A = np.diag(np.diag(A))
        np.fill_diagonal(A[1:],b)
        np.fill_diagonal(A[:,1:], c)
        A = (1/2)*(A + np.transpose(A))
        """
        A = np.zeros((n,n)).astype(np.float32)
        b = np.random.random_sample((n,1)).astype(np.float32)
        c = np.random.random_sample((n,1)).astype(np.float32)
        np.fill_diagonal(A[1:],b)
        np.fill_diagonal(A[:,1:], c)
        A = (1/2)*(A + np.transpose(A))
        sumrows = np.sum(A,axis=1)
        
        for i in range(n):
            th = np.random.uniform(sumrows[i],sumrows[i]+1) #satisify Gershgorin's
            A[i][i] = th
        
            
        d,v = np.linalg.eig(A)
        d = sorted(d, key=abs)
        train_A.append(A)
        train_eval.append(d)
        
    train_A = np.array(train_A)
    train_eval = np.array(train_eval)
    
    return train_A, train_eval
        

def l2_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(K.abs(y_true-y_pred))))

def l2_acc(y_true, y_pred):
    return 1-K.sqrt(K.sum(K.square(K.abs(y_true-y_pred))))

if __name__ == '__main__':
    
    run = return_params()
    
    #below kinda works
    input1 = Input(shape=(run.n,run.n,1))
    
    #the CNN architecture, following https://arxiv.org/pdf/1801.05733.pdf

    xin = Conv2D(16, kernel_size=(2,2), activation = "relu", kernel_regularizer=l2(0.0001), padding="same", input_shape=(run.n,1,1))(input1)
    print("shape of xin is ", xin.shape)
    

    x1 = Conv2D(16, kernel_size=(2,2), activation="relu", kernel_regularizer=l2(0.0001), padding="same")(xin)
    
    print("shape of x1 is ", x1.shape)
    

    x2 = AveragePooling2D(pool_size=(2,2))(x1)
    print("shape of x2 is ", x2.shape)

    x3 = Flatten()(x2) #comment out for tridiag test
    
    x4 = Dense(256, activation="linear")(x3)
    print("shape of x4 is ", x4.shape)
    
    xtmp1 = Dropout(0.5)(x4) #added for tridiag
    

    x5 = Dense(128, activation="linear")(xtmp1)
    
    xtmp = Dropout(0.0)(x5)#need dropout for tridiag, as it was overfitting without  

    
    x6 = Dense(run.n, activation="linear")(xtmp1)
    
    xtmp2 = Dropout(0.0)(x6) #tridiag
    

    xout = Reshape((run.n, 1, 1))(xtmp2) #tridiag
    
    model = Model(inputs=[input1], outputs=xout)
    print(model.summary())
    
    plot_model(model, to_file="super_eigen_network.png", show_shapes=True)
    opt = Adam(lr=1e-3, amsgrad=True)

    model.compile(loss=l2_loss, metrics=[l2_acc], optimizer=opt)
    

    A,d = get_tridiag_eigensystems(run.n,run.numtrain) #for a real symmetric tridiagonal matrix
    
    d = d.reshape(d.shape[0], run.n, 1, 1)
    A = A.reshape(A.shape[0], run.n, run.n, 1)
    

    history = model.fit([A], d, validation_split= 0.20, epochs=50, batch_size=1)
    print(history.history.keys())
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['l2_acc'])
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Training Accuracy'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_l2_acc'])
    plt.xlabel('Epoch')
    plt.legend(['Validation Loss', 'Validation Accuracy'], loc='upper left')
    plt.show()
    

    

    Atest, dtest = get_tridiag_eigensystems(run.n, 1)
    dtest = dtest.reshape(dtest.shape[0], run.n, 1, 1)
    Atest = Atest.reshape(Atest.shape[0], run.n, run.n, 1)
    pred = model.predict([Atest])
    
    print("predicted eigenvalues are ")
    print(pred)
    print("true eigenvalues are ")
    print(dtest)
    pred = pred.reshape(run.n)
    dtest = dtest.reshape(run.n)
    
    plt.plot(pred, dtest, 'bo')
    plt.plot([0,1,2,3],'--r')
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.show()