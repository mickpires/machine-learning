from tensorflow import keras
import tensorflow as tf
import numpy as np

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
        z = inputs

        for layer_name, layer in self.hiddens.items():
            z = layer(z)
        output = self.output_layer(z)
        return output
    

    def compute_loss(self, predictions, inputs, dndx):
        return tf.reduce_mean(tf.square(predictions + inputs*dndx - inputs))
    

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

    def train_step(self,data):
        inputs, _ = data
        with tf.GradientTape() as tape:
            with tf.GradientTape as tape2:
                tape2.watch(inputs)
                outputs = self(inputs,training=True)
            dndx = tape2.gradient(outputs,inputs)
            loss = self.compute_loss(outputs,inputs,dndx)
        dldw = tape.gradient(loss,self.trainable.weights)
        self.apply_gradients()
        return {"loss":loss}