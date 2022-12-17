
from keras.engine.base_layer import regularizers
from keras.layers import InputLayer, Dense, LSTM, Input, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adam,Adamax,Nadam,Ftrl,Adadelta
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.backend import clear_session
from tensorflow.keras.losses import mean_absolute_percentage_error, huber,kld
from sklearn.model_selection import train_test_split

clear_session()

kernel_reg_1=tf.keras.regularizers.L2(0.1)

input_size=20
drop_frac0=0.05  
drop_frac1=0.0  

input1=Input(shape=(input_size,))
l1_out=Dense(135,activation="swish",kernel_initializer='glorot_uniform',)(input1) # kernel_initializer='lecun_normal'
l2_out=Dropout(drop_frac0)(l1_out)


l3_out=Dense(15,activation="swish",kernel_initializer='glorot_uniform',)(l2_out) #kernel_initializer='lecun_normal',
l4_out=Dropout(drop_frac1)(l3_out)

pred=Dense(1, activation="sigmoid",)(l4_out)

model = Model(inputs=input1, outputs=pred)
optimizer=Adamax(learning_rate=0.001,) #

model.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=["accuracy"])
    
    
    
    
    
if __learning__: 
    history = model.fit(X_train, y_train, epochs=100, batch_size=3, validation_data=(X_test, y_test),verbose=1,callbacks=callbacks)
