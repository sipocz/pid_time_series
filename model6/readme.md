
kernel_reg_1=tf.keras.regularizers.L2(0.1)

input_size=20


input1=Input(shape=(input_size,))
l1_out=Dense(_N1_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(input1) # kernel_initializer='lecun_normal'  # L1

#l2_out=Dropout(_drop1_)(l1_out)

#l3_out=Dense(_N2_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(l2_out) #kernel_initializer='lecun_normal',  # L2
#l4_out=Dropout(_drop2_)(l3_out)

l5_out=Dense(_N3_,activation="linear",kernel_initializer='glorot_uniform',name="encoded",kernel_regularizer=None)(l1_out) #kernel_initializer='lecun_normal',  # L3

#l7_out=Dense(_N2_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(l5_out) #kernel_initializer='lecun_normal',  # L4

l9_out=Dense(_N1_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(l5_out) #kernel_initializer='lecun_normal',  # L5




pred=Dense(input_size, activation="sigmoid",)(l9_out)

model = Model(inputs=input1, outputs=pred)
optimizer=Adamax(learning_rate=_lr_,) #

model.compile(loss='MAE',
    optimizer=optimizer,
    metrics=["MAE"])
