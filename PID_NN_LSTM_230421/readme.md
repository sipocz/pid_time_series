val accuracy : 0.9982 


'''python

_N1_=1100 #70  #700
_N2_=118 #12  #120
_lr_=0.001
_batch_size_=32
_drop1_=0.5
_drop2_=0.7
_epochs_=9500



from keras.engine.base_layer import regularizers
from keras.layers import InputLayer, Dense, LSTM, Input, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adam,Adamax,Nadam,Ftrl,Adadelta
import tensorflow as tf
#from tensorflow.keras.callbacks import ModelCheckpoint
from keras.backend import clear_session
#from tensorflow.keras.losses import mean_absolute_percentage_error, huber,kld
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler

clear_session()

kernel_reg_1=tf.keras.regularizers.L2(0.1)

input_size=20


input1=Input(shape=(input_size,3))
l1_out=LSTM(input_size*2,)(input1) 
l2_out=Dropout(_drop1_)(l1_out)


l3_out=Dense(_N2_,activation="swish",kernel_initializer='glorot_uniform',)(l2_out) #kernel_initializer='lecun_normal',
l4_out=Dropout(_drop2_)(l3_out)

pred=Dense(1, activation="sigmoid",)(l4_out)

model = Model(inputs=input1, outputs=pred)
optimizer=Adamax(learning_rate=_lr_,) #

model.compile(loss='binary_crossentropy',
    optimizer=optimizer,
        metrics=["accuracy"])



'''
