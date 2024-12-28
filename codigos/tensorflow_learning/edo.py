from tensorflow import keras
import tensorflow as tf
import numpy as np

class EDOModel(keras.Model): # aqui eu crio o modelo
    def __init__(self,denses= 100,activations='sigmoid',hiddens = 1,**kwargs):
        super().__init__(**kwargs)
        self.hiddens = {}
        if type(denses) != list and type(denses) != tuple: # aqui eu verifico se o que eu tô passando no argumento é uma lista ou somente uma str e configuro para que toda a rede seja igual ao single input
             denses = [denses for _ in range(hiddens)]
        if type(activations) != list and type(denses) != tuple: # mesma coisa do de cima
             activations = [activations for _ in range(hiddens)]
        
        for hidden in range(hiddens): # aqui eu generalizo uma maneira de criar varias camadas ocultas
             name = f'hidden{hidden+1}'
             self.hiddens[name] = keras.layers.Dense(units=denses[hidden],activation=activations[hidden],name=name)
        
        self.output_layer = keras.layers.Dense(1) # camada de saida
    
    def call(self,inputs): #ele chama logo aṕós de eu usar o fit
        a = inputs # aqui eu chamo que o input é igual a para ficar mais parecido com a literatura, mas o certo seria chamar de z

        for layer_name, layer in self.hiddens.items():
            a = layer(a) # aqui seria o sum_i w_{ij} z_i + b_j
        output = self.output_layer(a) # aqui seria a camada de saida
        return output
    

    def compute_loss(self, predictions, inputs, dydx): #aqui seria o calculo da função de custo
        return tf.reduce_mean(tf.square(dydx - inputs))
    
    def train_step(self,data): # aqui seria como que funciona o treinamento
        inputs, targets = data #separo entre inputs e targets
        with tf.GradientTape() as tape: #aqui é para eu aplicar o autograd. Ele observa quais operações matematicas vão acontecer dentro da identação
            with tf.GradientTape() as tape2: #mesma coisa
                tape2.watch(inputs) # aqui eu falo para o autograd observa as entradas
                predictions = self(inputs,training=True) #forward pass
            dydx =tape2.gradient(predictions,inputs) #calculo a derivada do modelo em relação a entrada que é necessário para calcular o custo
            loss= self.compute_loss(predictions,inputs,dydx) # calculo o custo
        gradients = tape.gradient(loss,self.trainable_variables) # calculo a derivada do modelo em relação aos pesos para ajustar o modelo
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables)) #retropropagation

        return {"loss":loss} #retorna a função de perda desta epoch