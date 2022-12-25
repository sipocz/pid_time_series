
teljes 3 napnyi teszt komplett autoencoder tanítása.
minmax scaler alkalmazásával
Model: "model"

`python
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 20)]              0         
                                                                 
 dense (Dense)               (None, 13)                273       
                                                                 
 encoded (Dense)             (None, 3)                 42        
                                                                 
 dense_1 (Dense)             (None, 13)                52        
                                                                 
 dense_2 (Dense)             (None, 20)                280       
                                                                 
=================================================================
Total params: 647
Trainable params: 647
Non-trainable params: 0
_________________________________________________________________
`
