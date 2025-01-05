from tensorflow import keras
import tensorflow as tf
import numpy as np

class EDOModel(keras.Model):
    def __init__(self,denses= 100,activations='sigmoid',**kwargs):
        super().__init__(**kwargs)
        self.hidden_layer = keras.layers.Dense(denses,activation=activations)
        self.output_layer = keras.layers.Dense(1)
    
    def call(self,inputs):
        z = inputs
        z = self.hidden_layer(z)
        output = self.output_layer(z)
        return output
    

    def compute_loss(self, n, x, dndx):
        return tf.reduce_mean(tf.square(n + x*dndx + (x+(1+3*x**2)/(1+x+x**3))*(1 + x*dndx) -(x**3 + 2*x+x**2*(1+3*x**2)/(1+x+x**3))))
    

    # def train_step(self,data):
    #     inputs,_ = data
    #     with tf.GradientTape() as tape:
    #         hidden_output = self(inputs,training=True)

    #         # pesos de entrada
    #         wji = self.hiddens['hidden1'].kernel
    #         vj = self.output_layer.kernel

    #         #soma ponderada das ativações
    #         zj = tf.matmul(inputs,wji)
    #         sigmaj_prime = tf.sigmoid(zj) * (1-tf.sigmoid(zj))

    #         deriv = tf.reduce_sum(vj * tf.transpose(wji)*tf.transpose(sigmaj_prime),axis=1)
    #         loss = self.compute_loss(hidden_output,inputs,deriv)
    #     gradients = tape.gradient(loss,self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

    #     return{"loss": loss}

    def train_step(self,data): # aqui fazer que nem está no artigo
        inputs, _ = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            outputs = self(inputs,training=True)
            dndx = tape.gradient(outputs,dndx)
            loss = self.compute_loss(outputs,inputs,dndx)
        gradients = tape.gradient(dndx,self.trainable_weights)
        self.apply_gradients