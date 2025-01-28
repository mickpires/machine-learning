from tensorflow import keras
import tensorflow as tf
import numpy as np



class EDOModel(keras.Model):
    def __init__(self,denses= 100,activations='tanh',**kwargs):
        super().__init__(**kwargs)
        self.hidden_layer = keras.layers.Dense(denses,activation=activations,dtype='float64',kernel_initializer='random_normal')
        self.output_layer = keras.layers.Dense(1,dtype='float64',kernel_initializer='random_normal')

    def call(self,inputs):
        z = inputs
        z = self.hidden_layer(z)
        output = self.output_layer(z)
        return output
    

    def compute_loss(self,x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            n = self(x)
            ci = 0
            trial = ci + x*n
        dtrial_dx = tape.gradient(trial,x)
        return tf.reduce_sum(tf.square(dtrial_dx + 1/5 *trial - tf.exp(-x/5)*tf.cos(x)))
    
    def train_step(self,data): # aqui fazer que nem está no artigo
        inputs, _ = data
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)
        gradients = tape.gradient(loss,self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

        return {"loss":loss}