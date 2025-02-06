from tensorflow import keras
import tensorflow as tf
import numpy as np
import sympy as sy



class EDOModel(keras.Model):
    def __init__(
            self,edo_eq:sy.Equality,
            dependent_variable:sy.Function,
            independent_variable:sy.Symbol,
            trial_solution,
            ci:list,
            denses= 100,
            activations='tanh',
            **kwargs):
        
        super().__init__(**kwargs)
        self.hidden_layer = keras.layers.Dense(denses,activation=activations,dtype='float64')
        self.output_layer = keras.layers.Dense(1,dtype='float64')
        self.edo_eq = edo_eq.lhs - edo_eq.rhs
        self.dependent_variable = dependent_variable
        self.trial_solution = trial_solution
        self.independent_variable = independent_variable
        self.ci = tf.constant(ci,dtype=tf.float64)

    def call(self,inputs):
        inputs = tf.cast(inputs,dtype=tf.float64)
        z = inputs
        z = self.hidden_layer(z)
        output = self.output_layer(z)
        return self.trial_solution(output,inputs,self.ci)
    
    def compute_loss(self,x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            trial = self(x)
        dtrial_dx = tape.gradient(trial,x)
        return tf.reduce_sum(
            tf.square(
                sy.lambdify((sy.Derivative(self.dependent_variable,self.independent_variable),self.dependent_variable),self.edo_eq,'tensorflow')(dtrial_dx,trial)
            ))
    
    def train_step(self,data):
        inputs, _ = data
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)
        gradients = tape.gradient(loss,self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

        return {"loss":loss}