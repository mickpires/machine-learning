from tensorflow import keras
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class EDOModel(keras.Model):
    def __init__(self,denses= 100,activations='sigmoid',hiddens = 1,**kwargs):
        super().__init__(**kwargs)
        self.hiddens = {}
        if type(denses) != list and type(denses) != tuple:
             denses = [denses for _ in range(hiddens)]
        if type(activations) != list and type(denses) != tuple:
             activations = [activations for _ in range(hiddens)]
        
        for hidden in range(hiddens):
             name = f'hidden{hidden+1}'
             self.hiddens[name] = keras.layers.Dense(units=denses[hidden],activation=activations[hidden],name=name)
        
        self.output_layer = keras.layers.Dense(1)
    
    def call(self,inputs):
        a = inputs

        for layer_name, layer in self.hiddens.items():
            a = layer(a)
        output = self.output_layer(a)
        return output
    

    def compute_loss(self, predictions, inputs, dydx):
        return tf.reduce_mean(tf.square(dydx - inputs))
    
    def train_step(self,data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(inputs)
                predictions = self(inputs,training=True)
            dydx =tape2.gradient(predictions,inputs)
            logging.info(dydx)
            loss= self.compute_loss(predictions,inputs,dydx)
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

        return {"loss":loss}