
teljes 3 napnyi teszt feldolgozása  
komplett autoencoder tanítása.
minmax scaler alkalmazásával.  



[wandb adatok itt:](https://wandb.ai/pid_status/pid_autoencoder/runs/3k5sclxx?workspace=user-sipoczlaszlo)  
[youtube](https://youtu.be/QqwuHo6WiiY)  

```python
model.summary()
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
```
