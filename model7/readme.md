 [wandb address](https://wandb.ai/pid_status/pid_autoencoder/runs/sl8olbno])  
 
 Parameters:
''
_N1_=13  
_N2_=5  
_N3_=3  
_N4_=5  
_N5_=13  

_lr_=0.0001  
_batch_size_=32  
_drop1_=0.0  
_drop2_=0.0  
_epochs_=5000  
_comment_="3 réteg:  20] 13 2 13 [20 "  


input_size=20


input1=Input(shape=(input_size,))

l1_out=Dense(_N1_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(input1) # kernel_initializer='lecun_normal'  # L1

#l2_out=Dense(_N2_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(l1_out) #kernel_initializer='lecun_normal',  # L2

l3_out=Dense(_N3_,activation="linear",kernel_initializer='glorot_uniform',name="encoded",kernel_regularizer=None)(l1_out) #kernel_initializer='lecun_normal',  # L3

#l4_out=Dense(_N4_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(l3_out) #kernel_initializer='lecun_normal',  # L4

l5_out=Dense(_N5_,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer=None)(l3_out) #kernel_initializer='lecun_normal',  # L5




pred=Dense(input_size, activation="sigmoid",)(l5_out)

model = Model(inputs=input1, outputs=pred)

optimizer=Adamax(learning_rate=_lr_,) 

model.compile(loss='MAE', optimizer=optimizer, metrics=["MAE"]) 
''
